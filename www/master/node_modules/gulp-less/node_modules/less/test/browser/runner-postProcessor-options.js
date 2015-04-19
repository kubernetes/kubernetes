var less = {};
less.postProcessor = function(styles) {
    return 'hr {height:50px;}\n' + styles;
};
