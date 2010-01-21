/*
bayes.js - naïve Bayesian classification in JavaScript

This module is a JavaScript port of Divmod Reverend, 
(c) 2003 Amir Bakhtiar <amir@divmod.org>

 <http://divmod.org/trac/wiki/DivmodReverend>

(c) 2007 Sam Angove <sam@rephrase.net>

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, 
version 2.1 ONLY.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
*/

function cmp(x, y) {
    if (x > y) return 1;
    else if (x == y) return 0;
    else return -1;
}
/*
Make an array unique. It'll go down in flames if you
use it on an array of complex values, so don't.
*/
if (typeof Array.prototype.unique == "undefined")
Array.prototype.unique = function() {
    var tmp = {};
    var newArray = [];
    for (var i=0; i<this.length; ++i) {
        if (tmp.hasOwnProperty(this[i]))
            continue;
        newArray.push(this[i]);
        tmp[this[i]] = true;
    }
    return newArray;
};

/*
The Robinson method for combining probabilities.
*/
function robinson(probArray) {
    /*
    Robinson Method
    ---------------
    P (matches) = 1 - prod(1-p)^(1/n)
    Q (doesn't) = 1 - prod(p)^(1/n)
    S           = (1 + (P-Q)/(P+Q)) / 2
    */
    var n = probArray.length;
    var P = 1, Q = 1;
    for (var i=0; i<n; ++i) {
        P *= 1 - probArray[i][1];
        Q *= probArray[i][1];
    }
    P = 1 - Math.pow(P, (1/n));
    Q = 1 - Math.pow(Q, (1/n));
    return (1 + (P-Q)/(P+Q)) / 2;
}

var Bayes = function() {
    this.corpus = {
        data: {},
        tokenCount: 0,
        trainCount: 0
    };
    this.pools = {};
    this.cache = {};
    this.dirty = false;
};
Bayes.prototype = {
    tokenize: function(s) {
        return s.toLowerCase().split(/\s+/);
    },
    combine: robinson,
    /*
    Was the filter trained on the text with ID `id`?
    */
    trainedOn: function(id) {
        for (var pool in this.pools) {
            if (!this.pools.hasOwnProperty(pool))
                continue;
            if (this.pools[pool].training[id])
                return true;
        }
        return false;
    },
    /*
    Get all pool names.
    */
    poolNames: function() {
        var poolName, poolNames = [];
        for (poolName in this.pools) if (this.pools.hasOwnProperty(poolName))
            poolNames.push(poolName);
        poolNames.sort();
        return poolNames;
    },
    
    /* Return an array of [token, count] tuples. */
    poolData: function(poolName) {
        var token, poolCounts = [];
        var pool = this.pools[poolName];
        for (token in pool.data) 
            if (pool.data.hasOwnProperty(token))
                poolCounts.push([token, pool.data[token]]);
        return poolCounts;
    },
    
    /* Create a new pool. */
    newPool: function(poolName) {
        this.pools[poolName] = {
            name: poolName,
            data: {},
            tokenCount: 0,
            training: [],
            trainCount: 0
        };
        this.dirty = true;
        return this.pools[poolName];
    },
    
    /* Delete a pool. */
    removePool: function(poolName) {
        delete this.pools[poolName];
        this.dirty = true;
    },
    
    /* Rename a pool. */
    renamePool: function(poolName, renameTo) {
        this.pools[renameTo] = this.pools[poolName];
        this.pools[renameTo].name = renameTo;
        delete this.pools[poolName];
        this.dirty = true;
    },
    
    /* Merge an existing pool into another. `sourcePoolName` is
    left intact. */
    mergePools: function(sourcePoolName, destPoolName) {
        var token, training;
        var sourcePool = this.pools[sourcePoolName];
        var destPool = this.pools[destPoolName];
        for (token in sourcePool.data) {
            if (!sourcePool.data.hasOwnProperty(token))
                continue;
            if (token in destPool.data)
                destPool.data[token] += sourcePool.data[token];
            else
                destPool.data[token] = sourcePool.data[token];
        }
        for (training in sourcePool.training)
            if (sourcePool.training.hasOwnProperty(training))
                destPool.training[training] = true;
        this.dirty = true;
    },
    
    /*
    Get all pool probabilities.
    */
    getPoolProbs: function() {        
        if (this.dirty) {
            var poolName;
            for (poolName in this.pools) {
                if (!this.pools.hasOwnProperty(poolName))
                    continue;
                this.cache[poolName] = this.calculateProbs(this.pools[poolName]);
            }
        }
        this.dirty = false;
        return this.cache;
    },
    /*
    Get the probabilites of the given tokens from this pool.
    */
    getTokenProbs: function(poolData, tokens) {
        var probs = [], token;
        for (var i=0; i<tokens.length; i++) {
            token = tokens[i];
            if (token in poolData) {
                probs.push([token, poolData[token]]);
            }
        }
        probs.sort(function(x,y) { return cmp(y[1], x[1]); });
        return probs;
    },
    
    /*
    Train the filter to recognize `item` as matching `poolName`.
    The `item` may be optionally identified by a unique tracking `id`.
    */
    train: function(poolName, item, id) {
        if (!(poolName in this.pools))
            this.newPool(poolName);
        
        var pool = this.pools[poolName];
        var tokens = this.tokenize(item);
        var token, count, i;
        for (i=0; i<tokens.length; ++i) {
            token = tokens[i];
            if (token in pool.data) {
                pool.data[token]++;
            } else {
                pool.data[token] = 1;
            }
            if (token in this.corpus.data) {
                this.corpus.data[token]++;
            } else {
                this.corpus.data[token] = 1;
            }
        }
        pool.trainCount += 1;
        pool.tokenCount += tokens.length;
        this.corpus.trainCount += 1;
        this.corpus.tokenCount += tokens.length;
        if (typeof id != "undefined")
            pool.training[id] = true;
        this.dirty = true;
    },
    
    /*
    Retract previous training to identify `item` with `poolName`.
    Accepts an optional tracking `id`.
    
    If the filter has not been trained on the item, the behaviour
    is undefined.
    */
    untrain: function(poolName, item, id) {
        var tokens = self.tokenize(item);
        var pool = this.pools[poolName];
    
        var token;
        for (var i=0; i<tokens.length; ++i) {
            token = tokens[length];
            
            if (pool.data[token]) {
                pool.data[token] -= 1;
            } else {
                delete pool.data[token];
            }
            pool.tokenCount  -= 1;
            
            if (this.corpus.data[token]) {
                this.corpus.data[token] -= 1;
            } else {
                delete this.corpus.data[token];
            }
            this.corpus.tokenCount -= 1;
        }
        this.corpus.trainCount += 1;
        pool.trainCount += 1;
        if (typeof id != "undefined")
            delete pool.training[id];
        this.dirty = true;
    },
    
    /*
    Get the probabilities that the given string matches
    any of the pools.
    
    Returns an array of [poolName, probability] arrays, where
    0.0 <= probability <= 1.0.
    */
    guess: function(s, withProbs) {
        var tokens = this.tokenize(s).unique();
        var pools = this.getPoolProbs();
        var res = [], row;
        var poolName, probs;
        for (poolName in pools) {
            if (!pools.hasOwnProperty(poolName))
                continue;
            probs = this.getTokenProbs(pools[poolName], tokens);

            if (probs.length) {
                row = [poolName, this.combine(probs, poolName)];
                if (withProbs)
                    row.push(probs);
                res.push(row);
            }
        }
        res.sort(function(x,y) { return cmp(y[1], x[1]); });
        return res;
    },
    
    /*
    Calculate probabilities for tokens in a given pool based on
    occurrences in this pool vs. occurrences in the corpus as a whole.
    */
    calculateProbs: function(pool) {
        var poolSize = pool.tokenCount;
        var notInPoolSize = Math.max(this.corpus.tokenCount - poolSize, 1);
    
        var token, timesInPool, timesNotInPool; 
        var postive, negative;
        var prob, probs = {};
    
        for (token in pool.data) {
            if (!pool.data.hasOwnProperty(token))
                continue;
    
            timesInPool = pool.data[token];
            timesNotInPool = this.corpus.data[token] - timesInPool;
            
            /*
            Strength of the token as an indicator that a given text matches
            this pool.
            */
            negative = Math.min(1.0, timesNotInPool/poolSize);
            positive = Math.min(1.0, timesInPool/notInPoolSize);
            prob = positive / (negative + positive);
            
            /* 
            Don't bother caching really weak indicators. 
            */
            if (Math.abs(prob-0.5) >= 0.1) {
                probs[token] = Math.max(0.0001, Math.min(0.9999, prob));
            }
        }    
        return probs;
    }
};