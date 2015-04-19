define(function(){
    /**
    * Count number of full steps.
    */
    function countSteps(val, step, overflow){
        val = Math.floor(val / step);

        if (overflow) {
            return val % overflow;
        }

        return val;
    }

    return countSteps;
});
