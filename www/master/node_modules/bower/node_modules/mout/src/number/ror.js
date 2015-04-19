define(function(){
    /**
     * Bitwise circular shift right
     * http://en.wikipedia.org/wiki/Circular_shift
     */
    function ror(val, shift){
        return (val >> shift) | (val << (32 - shift));
    }
    return ror;
});
