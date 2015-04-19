define(['./filter', '../function/makeIterator_'], function (filter, makeIterator) {

    /**
     * Inverse or collection/filter
     */
    function reject(list, iterator, thisObj) {
        iterator = makeIterator(iterator, thisObj);
        return filter(list, function(value, index, list) {
            return !iterator(value, index, list);
        }, thisObj);
    }

    return reject;

});
