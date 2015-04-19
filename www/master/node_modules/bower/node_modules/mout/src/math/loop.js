define(function(){
    /**
    * Loops value inside range.
    */
    function loop(val, min, max){
        return val < min? max : (val > max? min : val);
    }

    return loop;
});
