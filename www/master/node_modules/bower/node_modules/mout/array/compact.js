var filter = require('./filter');

    /**
     * Remove all null/undefined items from array.
     */
    function compact(arr) {
        return filter(arr, function(val){
            return (val != null);
        });
    }

    module.exports = compact;

