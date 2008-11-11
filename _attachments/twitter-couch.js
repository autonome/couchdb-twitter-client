function CouchDesign(db, name) {
  this.view = function(view, opts) {
    db.view(name+'/'+view, opts);
  };
};

function TwitterCouch(db, design, callback) {  
  var currentTwitterID = null;
  var host = "twitter.com";
  function getJSON(path, params, cb) {
    var url = "http://"+host+path+".json?callback=?";
    $.getJSON(url, params, cb);
  };

  function cheapJSONP(url) {
    var s = document.createElement('script');
    s.src = url;
    s.type='text/javascript';
    document.body.appendChild(s);
  };

  function uniqueValues(rows) {
    var values = [];
    var keyMap = {};
    $.each(rows, function(i, row) {
      var key = row.key.toString();
      if (!keyMap[key]) {
        values.push(row.value);
        keyMap[key] = true;
      }
    });
    return values;
  };

  function viewFriendsTimeline(userId, cb) {
    design.view('friendsTimeline',{
      startkey : [userId,{}],
      endkey : [userId],
      group :true,
      descending : true,
      count : 50,
      success : function(json){
        cb(uniqueValues(json.rows), currentTwitterID);
      }
    });
  };
  
  function viewUserWordCloud(userid, cb) {
    design.view('userWordCloud', {
      startkey : [userid],
      endkey : [userid,{}],
      group_level : 2,
      success : function(data) {
        var cloud = [];
        $.each(data.rows, function(i,row) {
          if (row.value > 2) cloud.push([row.key[1], row.value]);
        });
        cb(cloud.sort(function(a,b) {
          return b[1] - a[1];
        }));
      }
    });
  };
  
  function apiCallProceed(force) {
    var previousCall = $.cookies.get('twitter-last-call');
    var d  = new Date;
    var now = d.getTime();
    if (force || !previousCall) {
      $.cookies.set('twitter-last-call', now);
      return true;
    } else {
      if (now - previousCall > 1000 * 60 * 2) {
        $.cookies.set('twitter-last-call', now);
        return true;
      } else {
        return false;
      }
    }
  };

  function getTwitterID(cb) {
    // todo what about when they are not logged in?
    var cookieID = $.cookies.get('twitter-user-id');
    if (cookieID) {
      currentTwitterID = cookieID;
      cb(publicMethods);
    } else {
      // this is hackish to get around the broken twitter cache
      window.userInfo = function(data) {
        currentTwitterID = data[0].user.id;
        $.cookies.set('twitter-user-id', currentTwitterID)
        callback(publicMethods);
      };
      cheapJSONP("http://"+host+"/statuses/user_timeline.json?count=1&callback=userInfo");      
    }
  };
  
  function getUserTimeline(userid, cb) {
    getJSON("/statuses/user_timeline/"+userid, {count:200}, function(tweets) {
      var doc = {
        tweets : tweets,
        userTimeline : userid
      };
      db.saveDoc(doc, {success:cb});
    });
  };
  
  function getFriendsTimeline(cb, opts) {
    getJSON("/statuses/friends_timeline", opts, function(tweets) {
      if (tweets.length > 0) {
        var doc = {
          tweets : tweets,
          friendsTimelineOwner : currentTwitterID
        };
        db.saveDoc(doc, {success:function() {
          viewFriendsTimeline(currentTwitterID, cb);
        }});
      }
    });    
  };
  
  var publicMethods = {
    friendsTimeline : function(cb, force) {
      viewFriendsTimeline(currentTwitterID, function(storedTweets) {
        cb(storedTweets, currentTwitterID);
        if (apiCallProceed(force)) {
          var newestTweet = storedTweets[0];
          var opts = {};
          if (newestTweet) {
            opts.since_id = newestTweet.id;
          }
          getFriendsTimeline(cb, opts);
        }
      });
    },
    updateStatus : function(status) {
      // todo in_reply_to_status_id
      $.xdom.post('http://twitter.com/statuses/update.json',{status:status});        
    },
    userInfo : function(userid, cb) {
      userid = parseInt(userid);
      design.view('userTweets', {
        startkey : [userid,{}],
        reduce : false,
        count : 1,
        descending: true,
        success : function(view) {
          cb(view.rows[0].value.user);
        }
      });
    },
    userWordCloud : function(userid, cb) {
      userid = parseInt(userid);
      // check to see if we've got the users back catalog.
      design.view('userTimeline', {
        key : userid,
        success : function(view) {
          // fetch it if not
          if (view.rows.length > 0) {
            viewUserWordCloud(userid, cb);
          } else {
            getUserTimeline(userid, function() {
              viewUserWordCloud(userid, cb);
            });
          }
        }
      });
    }
  };
  
  getTwitterID(callback);
};