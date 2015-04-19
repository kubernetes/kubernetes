module("Human lang");

test("Test lang attribute must be present", function() {
    var rule = axs.AuditRules.getRule('humanLangMissing');

    // Remove the humanLang attribute from the qunit test page.
    var htmlElement = document.querySelector('html');
    
    var htmlLang = htmlElement.lang;
    htmlElement.lang = '';

    equal(rule.run().result, 
        axs.constants.AuditResult.FAIL);

    htmlElement.lang = 'en-US';

    equal(rule.run().result, 
        axs.constants.AuditResult.PASS);

    htmlElement.lang = htmlLang;
});
