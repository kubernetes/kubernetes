var randBool = require('./randBool');

    /**
     * Returns random bit (0 or 1)
     */
    function randomBit() {
        return randBool()? 1 : 0;
    }

    module.exports = randomBit;

