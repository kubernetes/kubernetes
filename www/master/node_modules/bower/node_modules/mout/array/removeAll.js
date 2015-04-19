var indexOf = require('./indexOf');

    /**
     * Remove all instances of an item from array.
     */
    function removeAll(arr, item){
        var idx = indexOf(arr, item);
        while (idx !== -1) {
            arr.splice(idx, 1);
            idx = indexOf(arr, item, idx);
        }
    }

    module.exports = removeAll;

