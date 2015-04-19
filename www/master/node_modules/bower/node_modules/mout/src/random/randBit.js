define(['./randBool'], function (randBool) {

    /**
     * Returns random bit (0 or 1)
     */
    function randomBit() {
        return randBool()? 1 : 0;
    }

    return randomBit;
});
