'use strict';

var Q = require('q');
var arrayRemove = require('./lib/arrayRemove');

function PThrottler(defaultConcurrency, types) {
    this._defaultConcurrency = typeof defaultConcurrency === 'number' ? defaultConcurrency : 10;

    // Initialize some needed properties
    this._queue = {};
    this._slots = types || {};
    this._executing = [];
}

// -----------------

PThrottler.prototype.enqueue = function (func, type) {
    var deferred = Q.defer();
    var types;
    var entry;

    type = type || '';
    types = Array.isArray(type) ? type : [type];
    entry = {
        func: func,
        types: types,
        deferred: deferred
    };

    // Add the entry to all the types queues
    types.forEach(function (type) {
        var queue = this._queue[type] = this._queue[type] || [];
        queue.push(entry);
    }, this);

    // Process the entry shortly later so that handlers can be attached to the returned promise
    Q.fcall(this._processEntry.bind(this, entry));

    return deferred.promise;
};

PThrottler.prototype.abort = function () {
    var promises;

    // Empty the whole queue
    Object.keys(this._queue).forEach(function (type) {
        this._queue[type] = [];
    }, this);

    // Wait for all pending functions to finish
    promises = this._executing.map(function (entry) {
        return entry.deferred.promise;
    });

    return Q.allSettled(promises)
    .then(function () {});  // Resolve with no value
};

// -----------------

PThrottler.prototype._processQueue = function (type) {
    var queue = this._queue[type];
    var length = queue ? queue.length : 0;
    var x;

    for (x = 0; x < length; ++x) {
        if (this._processEntry(queue[x])) {
            break;
        }
    }
};

PThrottler.prototype._processEntry = function (entry) {
    var allFree = entry.types.every(this._hasSlot, this);
    var promise;

    // If there is a free slot for every type
    if (allFree) {
        // For each type
        entry.types.forEach(function (type) {
            // Remove entry from the queue
            arrayRemove(this._queue[type], entry);
            // Take slot
            this._takeSlot(type);
        }, this);

        // Execute the function
        this._executing.push(entry);
        promise = entry.func();
        if (typeof promise.then === 'undefined') {
            promise = Q.resolve(promise);
        }

        promise.progress(entry.deferred.notify.bind(entry.deferred));

        promise.then(
            this._onFulfill.bind(this, entry, true),
            this._onFulfill.bind(this, entry, false)
        );
    }

    return allFree;
};


PThrottler.prototype._onFulfill = function (entry, ok, result) {
    // Resolve/reject the deferred based on success/error of the promise
    if (ok) {
        entry.deferred.resolve(result);
    } else {
        entry.deferred.reject(result);
    }

    // Remove it from the executing list
    arrayRemove(this._executing, entry);

    // Free up slots for every type
    entry.types.forEach(this._freeSlot, this);

    // Find candidates for the free slots of each type
    entry.types.forEach(this._processQueue, this);
};

PThrottler.prototype._hasSlot = function (type) {
    var freeSlots = this._slots[type];

    if (freeSlots == null) {
        freeSlots = this._defaultConcurrency;
    }

    return freeSlots > 0;
};

PThrottler.prototype._takeSlot = function (type) {
    if (this._slots[type] == null) {
        this._slots[type] = this._defaultConcurrency;
    } else if (!this._slots[type]) {
        throw new Error('No free slots');
    }

    // Decrement the free slots
    --this._slots[type];
};

PThrottler.prototype._freeSlot = function (type) {
    if (this._slots[type] != null) {
        ++this._slots[type];
    }
};

PThrottler.create = function (defaultConcurrency, types) {
    return new PThrottler(defaultConcurrency, types);
};

module.exports = PThrottler;
