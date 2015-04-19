define(['./difference', './slice'], function (difference, slice) {

    /**
     * Insert item into array if not already present.
     */
    function insert(arr, rest_items) {
        var diff = difference(slice(arguments, 1), arr);
        if (diff.length) {
            Array.prototype.push.apply(arr, diff);
        }
        return arr.length;
    }
    return insert;
});
