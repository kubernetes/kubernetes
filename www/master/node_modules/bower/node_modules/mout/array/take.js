

    /**
     * Iterates over a callback a set amount of times
     * returning the results
     */
    function take(n, callback, thisObj){
        var i = -1;
        var arr = [];
        if( !thisObj ){
            while(++i < n){
                arr[i] = callback(i, n);
            }
        } else {
            while(++i < n){
                arr[i] = callback.call(thisObj, i, n);
            }
        }
        return arr;
    }

    module.exports = take;


