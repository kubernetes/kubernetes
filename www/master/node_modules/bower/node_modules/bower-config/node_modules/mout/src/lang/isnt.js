define(['./is'], function (is) {

    /**
     * Check if both values are not identical/egal
     */
    function isnt(x, y){
        return !is(x, y);
    }

    return isnt;

});
