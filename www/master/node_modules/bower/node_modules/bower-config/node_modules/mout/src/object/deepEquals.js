define(['../lang/isObject', './equals'], function (isObject, equals) {

    function defaultCompare(a, b) {
        return a === b;
    }

    /**
     * Recursively checks for same properties and values.
     */
    function deepEquals(a, b, callback){
        callback = callback || defaultCompare;

        if (!isObject(a) || !isObject(b)) {
            return callback(a, b);
        }

        function compare(a, b){
            return deepEquals(a, b, callback);
        }

        return equals(a, b, compare);
    }

    return deepEquals;

});
