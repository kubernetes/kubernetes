define(['./is', './isObject', './isArray', '../object/equals', '../array/equals'], function (is, isObject, isArray, objEquals, arrEquals) {

    /**
     * Recursively checks for same properties and values.
     */
    function deepEquals(a, b, callback){
        callback = callback || is;

        var bothObjects = isObject(a) && isObject(b);
        var bothArrays = !bothObjects && isArray(a) && isArray(b);

        if (!bothObjects && !bothArrays) {
            return callback(a, b);
        }

        function compare(a, b){
            return deepEquals(a, b, callback);
        }

        var method = bothObjects ? objEquals : arrEquals;
        return method(a, b, compare);
    }

    return deepEquals;

});
