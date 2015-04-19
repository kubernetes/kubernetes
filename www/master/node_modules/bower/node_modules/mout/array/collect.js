var append = require('./append');
var makeIterator = require('../function/makeIterator_');

    /**
     * Maps the items in the array and concatenates the result arrays.
     */
    function collect(arr, callback, thisObj){
        callback = makeIterator(callback, thisObj);
        var results = [];
        if (arr == null) {
            return results;
        }

        var i = -1, len = arr.length;
        while (++i < len) {
            var value = callback(arr[i], i, arr);
            if (value != null) {
                append(results, value);
            }
        }

        return results;
    }

    module.exports = collect;


