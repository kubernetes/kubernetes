define(['./filter', '../function/makeIterator_'], function (filter, makeIterator) {

    /**
     * Object reject
     */
    function reject(obj, callback, thisObj) {
        callback = makeIterator(callback, thisObj);
        return filter(obj, function(value, index, obj) {
            return !callback(value, index, obj);
        }, thisObj);
    }

    return reject;

});
