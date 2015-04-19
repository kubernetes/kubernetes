var is = require('./is');

    /**
     * Check if both values are not identical/egal
     */
    function isnt(x, y){
        return !is(x, y);
    }

    module.exports = isnt;


