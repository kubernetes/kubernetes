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
goog.require('axs.utils');

/**
 * @type {axs.AuditRule.Spec}
 */
axs.AuditRule.specs.imagesWithoutAltText = {
    name: 'imagesWithoutAltText',
    heading: 'Images should have an alt attribute',
    url: 'https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_text_02--images-should-have-an-alt-attribute-unless-they-have-an-aria-role-of-presentation',
    severity: axs.constants.Severity.WARNING,
    relevantElementMatcher: function(element) {
        return axs.browserUtils.matchSelector(element, 'img') &&
            !axs.utils.isElementOrAncestorHidden(element);
    },
    test: function(image) {
        return (!image.hasAttribute('alt') && image.getAttribute('role') != 'presentation');
    },
    code: 'AX_TEXT_02'
};
