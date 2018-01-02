(defconst testsuite-dir
  (if load-file-name
      (file-name-directory load-file-name)
    ;; Fall back to default directory (in case of M-x eval-buffer)
    default-directory)
  "Directory of the test suite.")

(defconst govc-test-helper-path
  (concat (expand-file-name (concat testsuite-dir "/../../test/test_helper.bash"))))

(load (expand-file-name "../govc" testsuite-dir) nil :no-message)

(ert-deftest test-govc-url-parse ()
  (dolist (u '("root:vagrant@localhost:18443"
               "Administrator@vsphere.local:vagrant@localhost"
               "https://root:vagrant@localhost:18443/sdk"
               "https://Administrator@vsphere.local:vagrant@localhost/sdk"))
    (should (equal u (url-recreate-url (govc-url-parse u))))))

(ert-deftest test-govc-session-set-url ()
  (should (equal govc-session-insecure nil))
  (should (equal govc-session-datacenter nil))
  (with-temp-buffer
    (govc-session-set-url "vc.example.com?insecure=true&datacenter=foo&ignored=true")
    (should (equal govc-session-insecure "true"))
    (should (equal govc-session-datacenter "foo"))
    (should (equal govc-session-datastore nil))))

(ert-deftest test-govc-copy-environment ()
  (let ((process-environment)
        (govc-session-url "vc.example.com")
        (govc-session-insecure "false")
        (govc-session-datacenter "dc1")
        (govc-session-datastore "ds1")
        (govc-session-network "net1"))
    (govc-export-environment '-)
    (dolist (e govc-environment-map)
      (should (equal nil (getenv (car e)))))
    (govc-export-environment (universal-argument))
    (dolist (e govc-environment-map)
      (should (not (equal nil (getenv (car e))))))))

(defun govc-test-env ()
  (let ((url (getenv "GOVC_TEST_URL")))
    (unless url
      (ert-skip "env GOVC_TEST_URL not set"))
    (setq govc-session-url url
          govc-session-insecure "true")))

(defun govc-test-helper (arg)
  (shell-command-to-string (format "bash -c \"source %s; %s\"" govc-test-helper-path arg)))

(defun govc-test-new-vm ()
  (s-trim-right (govc-test-helper "new_empty_vm")))

(defun govc-test-new-id ()
  (s-trim-right (govc-test-helper "new_id")))

(defun govc-test-teardown ()
  (ignore-errors
    (govc-test-helper "teardown")))

(ert-deftest test-govc-vm-info ()
  (govc-test-env)
  (unwind-protect
      (let ((id (govc-test-new-vm)))
        (govc-json-info "vm.info" (list id))
        (with-current-buffer "*govc-json*"
          (goto-char (point-min))
          (let ((data (json-read)))
            (should (= (length data) 1))
            (should (cdr (assq 'VirtualMachines data)))))

        (govc-json-info "vm.info" (list "ENOENT"))
        (with-current-buffer "*govc-json*"
          (goto-char (point-min))
          (let ((data (json-read)))
            (should (= (length data) 1))
            (should (not (cdr (assq 'VirtualMachines data))))))

        (let ((govc-args (list id))
              (len1)
              (len2))
          (setq len1 (length (govc-vm-extra-config)))
          (should (>= len1 1))
          (govc "vm.change" "-vm" id
                "-e" "govc-test-one=1"
                "-e" "govc-test-two:2.2=2"
                ;; test that we don't choke on \n
                "-e" "foo=bar
baz")
          (setq len2 (length (govc-vm-extra-config)))

          (should (= (- len2 len1) 3)))

        (let ((govc-filter "*"))
          (should (>= (length (govc-vm-info)) 1)))

        (let ((govc-filter "ENOENT"))
          (should (= (length (govc-vm-info)) 0)))

        (govc-vm-screen id))
    (govc-test-teardown)))

(ert-deftest test-govc-datastore-ls-entries ()
  (govc-test-env)
  (unwind-protect
      (let ((id (govc-test-new-id)))
        (should (>= (length (govc-datastore-ls-entries)) 1))

        (let ((govc-filter (concat id "/")))
          (should-error (govc-datastore-ls-entries))
          (govc "datastore.mkdir" id)
          (should (= (length (govc-datastore-ls-entries)) 0))
          (dotimes (i 3)
            (govc "datastore.mkdir" (format "%s/dir %d" id i)))
          (let ((entries (govc-datastore-ls-entries)))
            (should (= (length entries) 3))
            (should (s-starts-with? (concat id "/dir ") (caar entries))))))
    (govc-test-teardown)))

(ert-deftest test-govc-pool-ls ()
  (govc-test-env)
  (unwind-protect
      (let* ((pools (govc-ls-pool))
             (num (length pools))
             (path (concat (car pools) "/" (govc-test-new-id))))
        (should (>= num 1))
        (message "%d existing pools [%S]" num pools)
        (govc "pool.create" path)
        (setq pools (govc-ls-pool))
        (govc-pool-destroy path)
        (should (= (- (length pools) num) 1)))
    (govc-test-teardown)))

(ert-deftest test-govc-about ()
  (govc-test-env)
  (govc "about"))
