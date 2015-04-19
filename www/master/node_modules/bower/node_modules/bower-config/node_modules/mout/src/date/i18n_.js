define(['../object/mixIn', './i18n/en-US'], function(mixIn, enUS){

    // we also use mixIn to make sure we don't affect the original locale
    var activeLocale = mixIn({}, enUS, {
        // we expose a "set" method to allow overriding the global locale
        set : function(localeData){
            mixIn(activeLocale, localeData);
        }
    });

    return activeLocale;

});
