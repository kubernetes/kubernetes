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

goog.require('axs.constants');
goog.require('axs.browserUtils');

goog.provide('axs.AuditRule');
goog.provide('axs.AuditRule.Spec');

/**
 * @constructor
 * @param {axs.AuditRule.Spec} spec
 */
axs.AuditRule = function(spec) {
    var isValid = true;
    var missingFields = [];
    for (var i = 0; i < axs.AuditRule.requiredFields.length; i++) {
        var requiredField = axs.AuditRule.requiredFields[i];
        if (!(requiredField in spec)) {
            isValid = false;
            missingFields.push(requiredField);
        }
    }
    if (!isValid) {
        throw 'Invalid spec; the following fields were not specified: ' + missingFields.join(', ') +
              '\n' + JSON.stringify(spec);
    }

    /** @type {string} */
    this.name = spec.name;

    /** @type {axs.constants.Severity} */
    this.severity = spec.severity;

    /**
     * Is this element relevant to this audit?
     * @param {Element} element A potential audit candidate.
     * @return {boolean} true if this element is relevant to this audit.
     */
    this.relevantElementMatcher_ = spec.relevantElementMatcher;

    /**
     * The actual audit function.
     * @param {Element} element The element under test.
     * @param {Object=} config
     * @return {boolean} true if this audit finds a problem.
     */
    this.test_ = spec.test;

    /** @type {string} */
    this.code = spec.code;

    /** @type {string} */
    this.heading = spec.heading || '';

    /** @type {string} */
    this.url = spec.url || '';

    /** @type {boolean} */
    this.requiresConsoleAPI = !!spec['opt_requiresConsoleAPI'];
};

/** @typedef {{ name: string,
 *              heading: string,
 *              url: string,
 *              severity: axs.constants.Severity,
 *              relevantElementMatcher: function(Element): boolean,
 *              test: function(Element, Object=): boolean,
 *              code: string,
 *              opt_requiresConsoleAPI: boolean }} */
axs.AuditRule.SpecWithConsoleAPI;

/** @typedef {{ name: string,
 *              heading: string,
 *              url: string,
 *              severity: axs.constants.Severity,
 *              relevantElementMatcher: function(Element): boolean,
 *              test: function(Element, Object=): boolean,
 *              code: string }} */
axs.AuditRule.SpecWithoutConsoleAPI;

/** @typedef {(axs.AuditRule.SpecWithConsoleAPI|axs.AuditRule.SpecWithoutConsoleAPI)} */
axs.AuditRule.Spec;

/**
 * @const
 */
axs.AuditRule.requiredFields =
    [ 'name', 'severity', 'relevantElementMatcher', 'test', 'code', 'heading' ];


/**
 * The return value for a non-applicable audit result.
 *
 * @type {{result: string}}
 * @const
 */
axs.AuditRule.NOT_APPLICABLE = { result: axs.constants.AuditResult.NA };

/**
 * Add the given element to the given array.  This is abstracted so that subclasses
 * can modify the element value as necessary.
 * @param {Array.<Element>} elements
 * @param {Element} element
 */
axs.AuditRule.prototype.addElement = function(elements, element) {
    elements.push(element);
};

/**
 * Recursively collect elements which match |matcher| into |collection|,
 * starting at |node|.
 * @param {Node} node
 * @param {function(Element): boolean} matcher
 * @param {Array.<Element>} collection
 * @param {ShadowRoot=} opt_shadowRoot The nearest ShadowRoot ancestor, if any.
 */
axs.AuditRule.collectMatchingElements = function(node, matcher, collection,
                                                 opt_shadowRoot) {
    if (node.nodeType == Node.ELEMENT_NODE)
        var element = /** @type {Element} */ (node);

    if (element && matcher.call(null, element))
        collection.push(element);

    // Descend into node:
    // If it has a ShadowRoot, ignore all child elements - these will be picked
    // up by the <content> or <shadow> elements. Descend straight into the
    // ShadowRoot.
    if (element) {
        var shadowRoot = element.shadowRoot || element.webkitShadowRoot;
        if (shadowRoot) {
            axs.AuditRule.collectMatchingElements(shadowRoot,
                                                  matcher,
                                                  collection,
                                                  shadowRoot);
            return;
        }
    }

    // If it is a <content> element, descend into distributed elements - descend
    // into distributed elements - these are elements from outside the shadow
    // root which are rendered inside the shadow DOM.
    if (element && element.localName == 'content') {
        var content = /** @type {HTMLContentElement} */ (element);
        var distributedNodes = content.getDistributedNodes();
        for (var i = 0; i < distributedNodes.length; i++) {
            axs.AuditRule.collectMatchingElements(distributedNodes[i],
                                                  matcher,
                                                  collection,
                                                  opt_shadowRoot);
        }
        return;
    }

    // If it is a <shadow> element, descend into the olderShadowRoot of the
    // current ShadowRoot.
    if (element && element.localName == 'shadow') {
        var shadow = /** @type {HTMLShadowElement} */ (element);
        if (!opt_shadowRoot) {
            console.warn('ShadowRoot not provided for', element);
        } else {
            var olderShadowRoot = opt_shadowRoot.olderShadowRoot ||
                                  shadow.olderShadowRoot;
            if (olderShadowRoot) {
                axs.AuditRule.collectMatchingElements(olderShadowRoot,
                                                      matcher,
                                                      collection,
                                                      olderShadowRoot);
            }
        }
        return;
    }

    // If it is neither the parent of a ShadowRoot, a <content> element, nor
    // a <shadow> element recurse normally.
    var child = node.firstChild;
    while (child != null) {
        axs.AuditRule.collectMatchingElements(child,
                                              matcher,
                                              collection,
                                              opt_shadowRoot);
        child = child.nextSibling;
    }
}

/**
 * @param {Object} options
 *     Optional named parameters:
 *     ignoreSelectors: Selectors for parts of the page to ignore for this rule.
 *     scope: The scope in which the element selector should run.
 *         Defaults to `document`.
 *     maxResults: The maximum number of results to collect. If more than this
 *         number of results is found, 'resultsTruncated' is set to true in the
 *         returned object. If this is null or undefined, all results will be
 *         returned.
 *     config: Any per-rule configuration.
 * @return {?Object.<string, (axs.constants.AuditResult|?Array.<Element>|boolean)>}
 */
axs.AuditRule.prototype.run = function(options) {
    options = options || {};
    var ignoreSelectors = 'ignoreSelectors' in options ? options['ignoreSelectors'] : [];
    var scope = 'scope' in options ? options['scope'] : document;
    var maxResults = 'maxResults' in options ? options['maxResults'] : null;

    var relevantElements = [];
    axs.AuditRule.collectMatchingElements(scope, this.relevantElementMatcher_, relevantElements);

    var failingElements = [];

    function ignored(element) {
        for (var i = 0; i < ignoreSelectors.length; i++) {
          if (axs.browserUtils.matchSelector(element, ignoreSelectors[i]))
            return true;
        }
        return false;
    }

    if (!relevantElements.length)
        return { result: axs.constants.AuditResult.NA };
    for (var i = 0; i < relevantElements.length; i++) {
        if (maxResults != null && failingElements.length >= maxResults)
            break;
        var element = relevantElements[i];
        if (!ignored(element) && this.test_(element, options.config))
            this.addElement(failingElements, element);
    }
    var result = failingElements.length ? axs.constants.AuditResult.FAIL : axs.constants.AuditResult.PASS;
    var results = { result: result, elements: failingElements};
    if (i < relevantElements.length)
        results['resultsTruncated'] = true;

    return results;
};

axs.AuditRule.specs = {};
