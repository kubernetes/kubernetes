define(['./findLastIndex'], function (findLastIndex) {

    /**
     * Returns last item that matches criteria
     */
    function findLast(arr, iterator, thisObj){
        var idx = findLastIndex(arr, iterator, thisObj);
        return idx >= 0? arr[idx] : void(0);
    }

    return findLast;

});
