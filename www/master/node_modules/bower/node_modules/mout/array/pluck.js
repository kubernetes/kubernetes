var map = require('./map');

    /**
     * Extract a list of property values.
     */
    function pluck(arr, propName){
        return map(arr, propName);
    }

    module.exports = pluck;


