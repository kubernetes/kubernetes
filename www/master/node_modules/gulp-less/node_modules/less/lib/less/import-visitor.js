(function (tree) {
    tree.importVisitor = function(importer, finish, evalEnv, onceFileDetectionMap, recursionDetector) {
        this._visitor = new tree.visitor(this);
        this._importer = importer;
        this._finish = finish;
        this.env = evalEnv || new tree.evalEnv();
        this.importCount = 0;
        this.onceFileDetectionMap = onceFileDetectionMap || {};
        this.recursionDetector = {};
        if (recursionDetector) {
            for(var fullFilename in recursionDetector) {
                if (recursionDetector.hasOwnProperty(fullFilename)) {
                    this.recursionDetector[fullFilename] = true;
                }
            }
        }
    };

    tree.importVisitor.prototype = {
        isReplacing: true,
        run: function (root) {
            var error;
            try {
                // process the contents
                this._visitor.visit(root);
            }
            catch(e) {
                error = e;
            }

            this.isFinished = true;

            if (this.importCount === 0) {
                this._finish(error);
            }
        },
        visitImport: function (importNode, visitArgs) {
            var importVisitor = this,
                evaldImportNode,
                inlineCSS = importNode.options.inline;

            if (!importNode.css || inlineCSS) {

                try {
                    evaldImportNode = importNode.evalForImport(this.env);
                } catch(e){
                    if (!e.filename) { e.index = importNode.index; e.filename = importNode.currentFileInfo.filename; }
                    // attempt to eval properly and treat as css
                    importNode.css = true;
                    // if that fails, this error will be thrown
                    importNode.error = e;
                }

                if (evaldImportNode && (!evaldImportNode.css || inlineCSS)) {
                    importNode = evaldImportNode;
                    this.importCount++;
                    var env = new tree.evalEnv(this.env, this.env.frames.slice(0));

                    if (importNode.options.multiple) {
                        env.importMultiple = true;
                    }

                    this._importer.push(importNode.getPath(), importNode.currentFileInfo, importNode.options, function (e, root, importedAtRoot, fullPath) {
                        if (e && !e.filename) {
                            e.index = importNode.index; e.filename = importNode.currentFileInfo.filename;
                        }

                        var duplicateImport = importedAtRoot || fullPath in importVisitor.recursionDetector;
                        if (!env.importMultiple) {
                            if (duplicateImport) {
                                importNode.skip = true;
                            } else {
                                importNode.skip = function() {
                                    if (fullPath in importVisitor.onceFileDetectionMap) {
                                        return true;
                                    }
                                    importVisitor.onceFileDetectionMap[fullPath] = true;
                                    return false;
                                };
                            }
                        }

                        var subFinish = function(e) {
                            importVisitor.importCount--;

                            if (importVisitor.importCount === 0 && importVisitor.isFinished) {
                                importVisitor._finish(e);
                            }
                        };

                        if (root) {
                            importNode.root = root;
                            importNode.importedFilename = fullPath;

                            if (!inlineCSS && (env.importMultiple || !duplicateImport)) {
                                importVisitor.recursionDetector[fullPath] = true;
                                new(tree.importVisitor)(importVisitor._importer, subFinish, env, importVisitor.onceFileDetectionMap, importVisitor.recursionDetector)
                                    .run(root);
                                return;
                            }
                        }

                        subFinish();
                    });
                }
            }
            visitArgs.visitDeeper = false;
            return importNode;
        },
        visitRule: function (ruleNode, visitArgs) {
            visitArgs.visitDeeper = false;
            return ruleNode;
        },
        visitDirective: function (directiveNode, visitArgs) {
            this.env.frames.unshift(directiveNode);
            return directiveNode;
        },
        visitDirectiveOut: function (directiveNode) {
            this.env.frames.shift();
        },
        visitMixinDefinition: function (mixinDefinitionNode, visitArgs) {
            this.env.frames.unshift(mixinDefinitionNode);
            return mixinDefinitionNode;
        },
        visitMixinDefinitionOut: function (mixinDefinitionNode) {
            this.env.frames.shift();
        },
        visitRuleset: function (rulesetNode, visitArgs) {
            this.env.frames.unshift(rulesetNode);
            return rulesetNode;
        },
        visitRulesetOut: function (rulesetNode) {
            this.env.frames.shift();
        },
        visitMedia: function (mediaNode, visitArgs) {
            this.env.frames.unshift(mediaNode.rules[0]);
            return mediaNode;
        },
        visitMediaOut: function (mediaNode) {
            this.env.frames.shift();
        }
    };

})(require('./tree'));