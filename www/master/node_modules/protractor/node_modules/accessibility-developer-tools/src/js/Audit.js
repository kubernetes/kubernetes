// Copyright 2012 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

goog.require('axs.AuditResults');
goog.require('axs.AuditRules');
goog.require('axs.utils');

goog.provide('axs.Audit');
goog.provide('axs.AuditConfiguration');

/**
 * Object to hold configuration for an Audit run.
 * @constructor
 */
axs.AuditConfiguration = function() {
    /**
     * Dictionary of { audit rule name : { rules } } where rules is a dictionary
     * of { rule type : value }.
     * Possible rule types:
     * - ignore: value is a list of CSS selectors representing parts of the page
     *           which can be ignored for this audit rule.
     * - config: value is an object containing configuration for this audit
     *           rule. It will be passed to the test() method.
     * @type {Object}
     * @private
     */
    this.rules_ = {};

    /**
     * The "start point" for the audit: the element which contains the portion of
     * the page which should be audited.
     * If null, the document will be used as the scope.
     * @type {?Element}
     */
    this.scope = null;

    /**
     * A list of rule names representing the audit rules to be run. If this is
     * empty or |null|, all audit rules will be run.
     * @type {Array.<String>}
     */
    this.auditRulesToRun = null;

    /**
     * A list of rule names representing the audit rules which should not be run.
     * If this is empty or |null|, all audit rules will be run.
     * @type {Array.<String>}
     */
    this.auditRulesToIgnore = null;

    /**
     * The maximum number of results to collect for each audit rule. If more
     * than this number of results is found, 'resultsTruncated' is set to true
     * in the result object. If this is null, all results will be returned.
     */
    this.maxResults = null;

    /**
     * Whether this audit run can use the console API.
     * @type {boolean}
     */
    this.withConsoleApi = false;

    /**
     * Do we want to show a warning that there are audit rules which are not supported in this configuration?
     * @type {boolean}
     */
    this.showUnsupportedRulesWarning = true;

    goog.exportProperty(this, 'scope', this.scope);
    goog.exportProperty(this, 'auditRulesToRun', this.auditRulesToRun);
    goog.exportProperty(this, 'auditRulesToIgnore', this.auditRulesToIgnore);
    goog.exportProperty(this, 'withConsoleApi', this.withConsoleApi);
    goog.exportProperty(this, 'showUnsupportedRulesWarning', this.showUnsupportedRulesWarning );
};
goog.exportSymbol('axs.AuditConfiguration', axs.AuditConfiguration);

axs.AuditConfiguration.prototype = {
    /**
     * Add the given selectors to the ignore list for the given audit rule.
     * @param {string} auditRuleName The name of the audit rule
     * @param {Array.<string>} selectors Query selectors to match nodes to
     *     ignore
     */
    ignoreSelectors: function(auditRuleName, selectors) {
        if (!(auditRuleName in this.rules_))
            this.rules_[auditRuleName] = {};
        if (!('ignore' in this.rules_[auditRuleName]))
            this.rules_[auditRuleName].ignore = [];
        Array.prototype.push.call(this.rules_[auditRuleName].ignore, selectors);
    },

    /**
     * Gets the selectors which have been added to the ignore list for the given
     * audit rule.
     * @param {string} auditRuleName The name of the audit rule
     * @return {Array.<string>} A list of query selector strings which match nodes
     * to be ignored for the given rule.
     */
    getIgnoreSelectors: function(auditRuleName) {
        if ((auditRuleName in this.rules_) &&
            ('ignore' in this.rules_[auditRuleName])) {
            return this.rules_[auditRuleName].ignore;
        }
        return [];
    },

    /**
     * Sets the user-specified severity for the given audit rule. This will
     * replace the default severity for that audit rule in the audit results.
     * @param {string} auditRuleName
     * @param {axs.constants.Severity} severity
     */
    setSeverity: function(auditRuleName, severity) {
        if (!(auditRuleName in this.rules_))
            this.rules_[auditRuleName] = {};
        this.rules_[auditRuleName].severity = severity;
    },

    getSeverity: function(auditRuleName) {
        if (!(auditRuleName in this.rules_))
            return null;
        if (!('severity' in this.rules_[auditRuleName]))
            return null
        return this.rules_[auditRuleName].severity;
    },

    /**
     * Sets the user-specified configuration for the given audit rule. This will
     * vary in structure from rule to rule; see individual rules for
     * configuration options.
     * @param {string} auditRuleName
     * @param {Object} config
     */
    setRuleConfig: function(auditRuleName, config) {
        if (!(auditRuleName in this.rules_))
            this.rules_[auditRuleName] = {};
        this.rules_[auditRuleName].config = config;
    },

    /**
     * Gets the user-specified configuration for the given audit rule.
     * @param {string} auditRuleName
     * @return {Object?} The configuration object for the given audit rule.
     */
    getRuleConfig: function(auditRuleName) {
        if (!(auditRuleName in this.rules_))
            return null;
        if (!('config' in this.rules_[auditRuleName]))
            return null;
        return this.rules_[auditRuleName].config;
    }
};
goog.exportProperty(axs.AuditConfiguration.prototype, 'ignoreSelectors',
                    axs.AuditConfiguration.prototype.ignoreSelectors);
goog.exportProperty(axs.AuditConfiguration.prototype, 'getIgnoreSelectors',
                    axs.AuditConfiguration.prototype.getIgnoreSelectors);

axs.Audit.unsupportedRulesWarningShown = false;

/**
 * Returns the rules that cannot run.
 * For example, if the current configuration requires the console API, these
 * consist of the rules that require it.
 * @param {axs.AuditConfiguration=} opt_configuration
 * @return {Array.<String>}  A list of rules that cannot be run
 */
axs.Audit.getRulesCannotRun = function(opt_configuration) {
    if(opt_configuration.withConsoleApi) {
        return [];
    }
    return Object.keys(axs.AuditRule.specs)
        .filter(function(key) {
            return axs.AuditRules.getRule(key).requiresConsoleAPI;
        })
        .map(function(key) {
            return axs.AuditRules.getRule(key).code;
        });
}

/**
 * Runs an audit with all of the audit rules.
 * @param {axs.AuditConfiguration=} opt_configuration
 * @return {Array.<Object>} Array of Object:
 *     {
 *       result,    // @type {axs.constants.AuditResult}
 *       elements,  // @type {Array.<Element>}
 *       rule       // @type {axs.AuditRule} - data only (name, severity, code)
 *     }
 */
axs.Audit.run = function(opt_configuration) {
    var configuration = opt_configuration || new axs.AuditConfiguration();
    var withConsoleApi = configuration.withConsoleApi;
    var results = [];

    var auditRules;
    if (configuration.auditRulesToRun &&
        configuration.auditRulesToRun.length > 0) {
        auditRules = configuration.auditRulesToRun;
    } else
        auditRules = Object.keys(axs.AuditRule.specs);

    if (configuration.auditRulesToIgnore) {
        for (var i = 0; i < configuration.auditRulesToIgnore.length; i++) {
            var auditRuleToIgnore = configuration.auditRulesToIgnore[i];
            if (auditRules.indexOf(auditRuleToIgnore) < 0)
                continue;
            auditRules.splice(auditRules.indexOf(auditRuleToIgnore), 1);
        }
    }

    if(!axs.Audit.unsupportedRulesWarningShown && configuration.showUnsupportedRulesWarning) {
        var unsupportedRules = axs.Audit.getRulesCannotRun(configuration);
        if(unsupportedRules.length > 0) {
            console.warn("Some rules cannot be checked using the axs.Audit.run() method call. Use the Chrome plugin to check these rules: " + unsupportedRules.join(", "));
            console.warn("To remove this message, pass an AuditConfiguration object to axs.Audit.run() and set configuration.showUnsupportedRulesWarning = false.");
        }

        axs.Audit.unsupportedRulesWarningShown = true;
    }

    for (var i = 0; i < auditRules.length; i++) {
        var auditRuleName = auditRules[i];
        var auditRule = axs.AuditRules.getRule(auditRuleName);
        if (!auditRule)
            continue; // Shouldn't happen, but fail silently if it does.
        if (auditRule.disabled)
            continue;
        if (!withConsoleApi && auditRule.requiresConsoleAPI)
            continue;

        var options = {};
        var ignoreSelectors = configuration.getIgnoreSelectors(auditRule.name);
        if (ignoreSelectors.length > 0 || configuration.scope)
            options['ignoreSelectors'] = ignoreSelectors;
        var ruleConfig = configuration.getRuleConfig(auditRule.name);
        if (ruleConfig != null)
            options['config' ] = ruleConfig;
        if (configuration.scope)
            options['scope'] = configuration.scope;
        if (configuration.maxResults)
            options['maxResults'] = configuration.maxResults;
        var result = auditRule.run.call(auditRule, options);
        var ruleValues = axs.utils.namedValues(auditRule);
        ruleValues.severity = configuration.getSeverity(auditRuleName) ||
                              ruleValues.severity;
        result.rule = ruleValues;
        results.push(result);
    }

    return results;
};
goog.exportSymbol('axs.Audit.run', axs.Audit.run);

/**
 * Create an AuditResults object citing failures and warnings, for use in
 * continuous builds.
 * @param {Array.<Object>} results The results returned from the audit run.
 * @return  {axs.AuditResults} a report of the audit results.
 */
axs.Audit.auditResults = function(results) {
    var auditResults = new axs.AuditResults();
    for (var i = 0; i < results.length; i++) {
        var result = results[i];
        if (result.result != axs.constants.AuditResult.FAIL)
            continue;

        if (result.rule.severity == axs.constants.Severity.SEVERE) {
            auditResults.addError(
                axs.Audit.accessibilityErrorMessage(result));
        } else {
            auditResults.addWarning(
                axs.Audit.accessibilityErrorMessage(result));
        }
    }
    return auditResults;
};
goog.exportSymbol('axs.Audit.auditResults', axs.Audit.auditResults);

/**
 * Create a report based on the results of an Audit.
 * @param {Array.<Object>} results The results returned from axs.Audit.run();
 * @param {?string} opt_url A URL to visit for more information.
 * @return {string} A report of the audit results.
 */
axs.Audit.createReport = function(results, opt_url) {
    var message = '*** Begin accessibility audit results ***';
    message += '\nAn accessibility audit found ';

    message += axs.Audit.auditResults(results).toString();

    if (opt_url) {
        message += '\nFor more information, please see ' ;
        message += opt_url;
    }

    message += '\n*** End accessibility audit results ***';
    return message;
};
goog.exportSymbol('axs.Audit.createReport', axs.Audit.createReport);

/**
 * Creates an error message for a given accessibility audit result object.
 * @param {Object.<string, (string|Array.<Element>)>} result The result
 *     object returned from the audit.
 * @return {string} An error message describing the failure and listing the
 *     query selectors for up to five elements which failed the audit rule.
 */
axs.Audit.accessibilityErrorMessage = function(result) {
    if (result.rule.severity == axs.constants.Severity.SEVERE)
        var message = 'Error: '
    else
        var message = 'Warning: '
    message += result.rule.code + ' (' + result.rule.heading +
        ') failed on the following ' +
        (result.elements.length == 1 ? 'element' : 'elements');

    if (result.elements.length == 1)
        message += ':'
    else {
        message += ' (1 - ' + Math.min(5, result.elements.length) +
                   ' of ' + result.elements.length + '):';
    }

    var maxElements = Math.min(result.elements.length, 5);
    for (var i = 0; i < maxElements; i++) {
        var element = result.elements[i];
        message += '\n';
        // Get query selector not browser independent. catch any errors and
        // default to simple tagName.
        try {
            message += axs.utils.getQuerySelectorText(element);
        } catch (err) {
            message += ' tagName:' + element.tagName;
            message += ' id:' + element.id;
        }
    }
    if (result.rule.url != '')
        message += '\nSee ' + result.rule.url + ' for more information.';
    return message;
};
goog.exportSymbol('axs.Audit.accessibilityErrorMessage', axs.Audit.accessibilityErrorMessage);
