define(['./indexOf'], function (indexOf) {

    /**
     * Combines an array with all the items of another.
     * Does not allow duplicates and is case and type sensitive.
     */
    function combine(arr1, arr2) {
        if (arr2 == null) {
            return arr1;
        }

        var i = -1, len = arr2.length;
        while (++i < len) {
            if (indexOf(arr1, arr2[i]) === -1) {
                arr1.push(arr2[i]);
            }
        }

        return arr1;
    }
    return combine;
});
