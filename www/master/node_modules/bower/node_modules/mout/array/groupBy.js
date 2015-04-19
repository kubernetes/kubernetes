var forEach = require('../array/forEach');
var identity = require('../function/identity');
var makeIterator = require('../function/makeIterator_');

    /**
     * Bucket the array values.
     */
    function groupBy(arr, categorize, thisObj) {
        if (categorize) {
            categorize = makeIterator(categorize, thisObj);
        } else {
            // Default to identity function.
            categorize = identity;
        }

        var buckets = {};
        forEach(arr, function(element) {
            var bucket = categorize(element);
            if (!(bucket in buckets)) {
                buckets[bucket] = [];
            }

            buckets[bucket].push(element);
        });

        return buckets;
    }

    module.exports = groupBy;

