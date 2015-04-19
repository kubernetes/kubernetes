var sort = require('./sort');
var makeIterator = require('../function/makeIterator_');

    /*
     * Sort array by the result of the callback
     */
    function sortBy(arr, callback, context){
        callback = makeIterator(callback, context);

        return sort(arr, function(a, b) {
            a = callback(a);
            b = callback(b);
            return (a < b) ? -1 : ((a > b) ? 1 : 0);
        });
    }

    module.exports = sortBy;


