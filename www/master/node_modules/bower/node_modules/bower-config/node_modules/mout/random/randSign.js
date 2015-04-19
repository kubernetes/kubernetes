var randBool = require('./randBool');

    /**
     * Returns random sign (-1 or 1)
     */
    function randomSign() {
        return randBool()? 1 : -1;
    }

    module.exports = randomSign;

