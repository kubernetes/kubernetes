var random = require('./random');

    /**
     * returns a random boolean value (true or false)
     */
    function randBool(){
        return random() >= 0.5;
    }

    module.exports = randBool;


