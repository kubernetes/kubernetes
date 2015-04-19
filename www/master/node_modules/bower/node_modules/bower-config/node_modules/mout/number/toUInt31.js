var MAX_INT = require('./MAX_INT');

    /**
     * "Convert" value into an 31-bit unsigned integer (since 1 bit is used for sign).
     * IMPORTANT: value wil wrap at 2^31, if negative will return 0.
     */
    function toUInt31(val){
        // we do not use lang/toNumber because of perf and also because it
        // doesn't break the functionality
        return (val <= 0)? 0 : (val > MAX_INT? ~~(val % (MAX_INT + 1)) : ~~val);
    }

    module.exports = toUInt31;


