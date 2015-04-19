var isObject = require('../lang/isObject');
var values = require('../object/values');
var arrMap = require('../array/map');
var makeIterator = require('../function/makeIterator_');

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

    module.exports = map;


