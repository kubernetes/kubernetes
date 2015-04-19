var arrMax = require('../array/max');
var values = require('./values');

    /**
     * Returns maximum value inside object.
     */
    function max(obj, compareFn) {
        return arrMax(values(obj), compareFn);
    }

    module.exports = max;

