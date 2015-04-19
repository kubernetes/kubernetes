define(['../lang/is', '../lang/isArray', './every'], function(is, isArray, every) {

    /**
     * Compares if both arrays have the same elements
     */
    function equals(a, b, callback){
        callback = callback || is;

        if (!isArray(a) || !isArray(b)) {
            return callback(a, b);
        }

        if (a.length !== b.length) {
            return false;
        }

        return every(a, makeCompare(callback), b);
    }

    function makeCompare(callback) {
        return function(value, i) {
            return i in this && callback(value, this[i]);
        };
    }

    return equals;

});
