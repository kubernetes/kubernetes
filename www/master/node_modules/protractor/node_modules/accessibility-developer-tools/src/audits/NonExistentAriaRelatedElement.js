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

goog.require('axs.AuditRule');
goog.require('axs.AuditRules');
goog.require('axs.constants.Severity');

/**
 * @type {axs.AuditRule.Spec}
 */
axs.AuditRule.specs.nonExistentAriaRelatedElement = {
    name: 'nonExistentAriaRelatedElement',
    heading: 'ARIA attributes which refer to other elements by ID should refer to elements which exist in the DOM',
    url: '',  // TODO(RickSBrown): talk to Alice about wiki for this (I don't think I can update?);
    severity: axs.constants.Severity.SEVERE,
    relevantElementMatcher: function(element) {
        var idrefTypes = ['idref', 'idref_list'];
        var idRefProps = axs.utils.getAriaPropertiesByValueType(idrefTypes);
        var selector = axs.utils.getSelectorForAriaProperties(idRefProps);
        return axs.browserUtils.matchSelector(element, selector);
    },
    test: function(element) {
        var idrefTypes = ['idref', 'idref_list'];
        var idRefProps = axs.utils.getAriaPropertiesByValueType(idrefTypes);
        var selector = axs.utils.getSelectorForAriaProperties(idRefProps);
        var selectors = selector.split(',');
        for (var i = 0, len = selectors.length; i < len; i++) {
            var nextSelector = selectors[i];
            if (axs.browserUtils.matchSelector(element, nextSelector)) {
                var propertyName = nextSelector.match(/aria-[^\]]+/)[0];
                var propertyValueText = element.getAttribute(propertyName);
                var propertyValue = axs.utils.getAriaPropertyValue(propertyName,
                                                                   propertyValueText,
                                                                   element);
                if (!propertyValue.valid)
                    return true;
            }
        }
        return false;
    },
    code: 'AX_ARIA_02'
};
