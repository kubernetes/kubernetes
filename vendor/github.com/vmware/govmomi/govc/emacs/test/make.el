#!/usr/bin/env emacs --script

(let ((current-directory (file-name-directory load-file-name)))
  (setq project-test-path (expand-file-name "." current-directory))
  (setq project-root-path (expand-file-name ".." current-directory)))

(add-to-list 'load-path project-root-path)
(add-to-list 'load-path project-test-path)

(require 'lisp-mnt)
(require 'govc)
(require 's)

(defun make-test ()
  (dolist (test-file (or argv (directory-files project-test-path t "-test.el$")))
    (load test-file nil t))
  (ert-run-tests-batch-and-exit t))

(defun govc-help ()
  "Summary of govc modes in markdown format."
  (interactive)
  (with-help-window (help-buffer) ; TODO: this turned into a mess, but does the job of generating README.md from govc.el
    (dolist (kind '(govc-mode govc-urls
                    govc-session-url govc-session-insecure govc-session-datacenter govc-session-datastore govc-session-network
                    tabulated-list host pool datastore datastore-ls vm device object metric))
      (let* ((name (if (boundp kind) (symbol-name kind) (format "govc-%s-mode" kind)))
             (map (if (equal 'govc-mode kind) 'govc-command-map (intern (concat name "-map"))))
             (doc (lambda (f &optional all)
                    (let* ((txt (if (functionp f) (documentation f t) (documentation-property f 'variable-documentation)))
                           (ix (if all (length txt) (s-index-of "." txt))))
                      (s-replace (format "\n\n\\\{%s\}" (concat name "-map")) ""
                                 (s-replace "'" "`" (substring txt 0 ix)))))))
        (princ (concat (s-repeat (if (and (boundp kind) (not (fboundp kind))) 3 2) "#") " " name "\n"))
        (princ (concat "\n" (funcall doc (intern name) t) "\n\n"))
        (when (boundp map)
          (princ (concat "### " (symbol-name map) "\n\n"))
          (princ "Keybinding     | Description\n")
          (princ "---------------|------------------------------------------------------------\n")
          (dolist (kl (govc-keymap-list (symbol-value map)))
            (let ((key (govc-key-description (car kl))))
              (princ (format "<kbd>%s</kbd>%s| %s\n" key (s-repeat (- 4 (length key)) " ") (funcall doc (nth 2 kl))))))
          (princ "\n"))))))

(defun make-docs ()
  (let ((commentary)
        (summary))
    (with-current-buffer (find-file-noselect (concat project-root-path "/govc.el"))
      (setq commentary (s-replace ";;; Commentary:" "" (lm-commentary))
            summary (lm-summary)))
    (let ((readme (find-file-noselect (concat project-root-path "/README.md"))))
      (with-current-buffer readme
        (erase-buffer)
        (govc-help)
        (with-current-buffer (help-buffer)
          (copy-to-buffer readme (point-min) (point-max)))
        (goto-char (point-min))
        (insert (concat "# govc.el\n\n" summary ".\n"))
        (insert (s-replace "'" "`" (replace-regexp-in-string ";; ?" "" commentary t t)))
        (save-buffer 0)))))
