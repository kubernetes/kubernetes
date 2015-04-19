define(function () {

    /**
     * Returns last element of array.
     */
    function last(arr){
        if (arr == null || arr.length < 1) {
            return undefined;
        }

        return arr[arr.length - 1];
    }

    return last;

});
