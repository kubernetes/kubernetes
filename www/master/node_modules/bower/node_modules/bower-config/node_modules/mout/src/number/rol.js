define(function(){
    /**
     * Bitwise circular shift left
     * http://en.wikipedia.org/wiki/Circular_shift
     */
    function rol(val, shift){
        return (val << shift) | (val >> (32 - shift));
    }
    return rol;
});
