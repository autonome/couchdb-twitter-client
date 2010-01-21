function(doc) {
  if (doc.mentionsOwner && doc.tweets) {
    doc.tweets.forEach(function(tweet) {
      if (tweet.id)
        emit([doc.mentionsOwner, tweet.id], tweet);    
    });
  }
};
