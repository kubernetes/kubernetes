define(['../lang/isObject', '../object/values', '../array/map', '../function/makeIterator_'], function (isObject, values, arrMap, makeIterator) {

    /**
     * Map collection values, returns Array.
     */
    function map(list, callback, thisObj) {
        callback = makeIterator(callback, thisObj);
        // list.length to check array-like object, if not array-like
        // we simply map all the object values
        if( isObject(list) && list.length == null ){
            list = values(list);
        }
        return arrMap(list, function (val, key, list) {
            return callback(val, key, list);
        });
    }

    return map;

});
