define(['./isNumber'], function (isNumber) {

    var global = this;

    /**
     * Check if value is finite
     */
    function isFinite(val){
        var is = false;
        if (typeof val === 'string' && val !== '') {
            is = global.isFinite( parseFloat(val) );
        } else if (isNumber(val)){
            // need to use isNumber because of Number constructor
            is = global.isFinite( val );
        }
        return is;
    }

    return isFinite;

});
