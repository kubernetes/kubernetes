/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package helmapplyset

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/labeler"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

const (
	// WebhookPort is the port for the webhook server
	WebhookPort = 8443

	// ControllerUserAgent is the user-agent string for the HelmApplySet controller
	ControllerUserAgent = "helm-applyset-controller"

	// ControllerServiceAccount is the service account name for the controller
	ControllerServiceAccount = "helm-applyset-controller"

	// DefaultTimeout is the default timeout for webhook requests
	DefaultTimeout = 10 * time.Second
)

var (
	runtimeScheme = runtime.NewScheme()
	codecs        = serializer.NewCodecFactory(runtimeScheme)
	deserializer  = codecs.UniversalDeserializer()
)

func init() {
	_ = v1.AddToScheme(runtimeScheme)
	_ = admissionv1.AddToScheme(runtimeScheme)
}

// WebhookServer handles admission webhook requests for ApplySet validation and mutation
type WebhookServer struct {
	server       *http.Server
	tlsConfig    *tls.Config
	logger       klog.Logger
	failOpen     bool // If true, allow requests on webhook errors
	controllerSA string
	controllerUA string
}

// NewWebhookServer creates a new admission webhook server
func NewWebhookServer(
	tlsConfig *tls.Config,
	logger klog.Logger,
	failOpen bool,
) *WebhookServer {
	mux := http.NewServeMux()

	server := &WebhookServer{
		tlsConfig:    tlsConfig,
		logger:       logger,
		failOpen:     failOpen,
		controllerSA: ControllerServiceAccount,
		controllerUA: ControllerUserAgent,
		server: &http.Server{
			Addr:         fmt.Sprintf(":%d", WebhookPort),
			Handler:      mux,
			TLSConfig:    tlsConfig,
			ReadTimeout:  DefaultTimeout,
			WriteTimeout: DefaultTimeout,
		},
	}

	// Register handlers
	mux.HandleFunc("/validate", server.handleValidate)
	mux.HandleFunc("/mutate", server.handleMutate)
	mux.HandleFunc("/healthz", server.handleHealth)
	mux.HandleFunc("/readyz", server.handleReady)

	return server
}

// Start starts the webhook server
func (s *WebhookServer) Start(ctx context.Context) error {
	logger := klog.FromContext(ctx)
	s.logger = logger

	s.logger.Info("Starting admission webhook server", "port", WebhookPort)

	// Start server in goroutine
	go func() {
		if err := s.server.ListenAndServeTLS("", ""); err != nil && err != http.ErrServerClosed {
			s.logger.Error(err, "Failed to start webhook server")
		}
	}()

	// Wait for context cancellation
	<-ctx.Done()
	s.logger.Info("Shutting down webhook server")

	// Graceful shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := s.server.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("failed to shutdown webhook server: %w", err)
	}

	return nil
}

// handleHealth handles health check requests
func (s *WebhookServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}

// handleReady handles readiness check requests
func (s *WebhookServer) handleReady(w http.ResponseWriter, r *http.Request) {
	// Check if TLS config is ready
	if s.tlsConfig == nil || len(s.tlsConfig.Certificates) == 0 {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte("not ready: TLS not configured"))
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ready"))
}

// handleValidate handles validating admission webhook requests
func (s *WebhookServer) handleValidate(w http.ResponseWriter, r *http.Request) {
	admissionReview, err := s.parseAdmissionReview(r)
	if err != nil {
		s.logger.Error(err, "Failed to parse admission review")
		if s.failOpen {
			s.writeAllowResponse(w, r, "failed to parse request, allowing due to fail-open policy")
			return
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate the object
	response := s.validateObject(admissionReview.Request)
	response.UID = admissionReview.Request.UID

	admissionReview.Response = response
	s.writeResponse(w, admissionReview)
}

// handleMutate handles mutating admission webhook requests
func (s *WebhookServer) handleMutate(w http.ResponseWriter, r *http.Request) {
	admissionReview, err := s.parseAdmissionReview(r)
	if err != nil {
		s.logger.Error(err, "Failed to parse admission review")
		if s.failOpen {
			s.writeAllowResponse(w, r, "failed to parse request, allowing due to fail-open policy")
			return
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Mutate the object
	response := s.mutateObject(admissionReview.Request)
	response.UID = admissionReview.Request.UID

	admissionReview.Response = response
	s.writeResponse(w, admissionReview)
}

// parseAdmissionReview extracts an AdmissionReview from an http.Request
func (s *WebhookServer) parseAdmissionReview(r *http.Request) (*admissionv1.AdmissionReview, error) {
	if r.Header.Get("Content-Type") != "application/json" {
		return nil, fmt.Errorf("Content-Type: %q should be %q", r.Header.Get("Content-Type"), "application/json")
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read request body: %w", err)
	}
	if len(body) == 0 {
		return nil, fmt.Errorf("admission request HTTP body is empty")
	}

	review := &admissionv1.AdmissionReview{}
	if _, _, err := deserializer.Decode(body, nil, review); err != nil {
		return nil, fmt.Errorf("could not deserialize incoming admission review: %w", err)
	}

	if review.Request == nil {
		return nil, fmt.Errorf("admission review can't be used: Request field is nil")
	}

	return review, nil
}

// validateObject validates an object based on its type
func (s *WebhookServer) validateObject(request *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
	logger := s.logger.WithValues(
		"kind", request.Kind.Kind,
		"name", request.Name,
		"namespace", request.Namespace,
		"operation", request.Operation,
	)

	// Only validate Secrets (for ApplySet parent objects)
	if request.Kind.Kind != "Secret" || request.Kind.Group != "" || request.Kind.Version != "v1" {
		return &admissionv1.AdmissionResponse{
			Allowed: true,
		}
	}

	// Decode the Secret object
	secret := &v1.Secret{}
	var obj runtime.Object
	if request.Operation == admissionv1.Delete {
		// For delete operations, use OldObject
		if request.OldObject.Raw == nil {
			return &admissionv1.AdmissionResponse{
				Allowed: true, // Allow if we can't validate
			}
		}
		obj = secret
		if _, _, err := deserializer.Decode(request.OldObject.Raw, nil, obj); err != nil {
			logger.Error(err, "Failed to decode Secret for validation")
			return &admissionv1.AdmissionResponse{
				Allowed: true, // Allow if we can't decode
			}
		}
	} else {
		// For create/update operations, use Object
		if request.Object.Raw == nil {
			return &admissionv1.AdmissionResponse{
				Allowed: true,
			}
		}
		obj = secret
		if _, _, err := deserializer.Decode(request.Object.Raw, nil, obj); err != nil {
			logger.Error(err, "Failed to decode Secret for validation")
			return &admissionv1.AdmissionResponse{
				Allowed: true,
			}
		}
	}

	// Check if this is an ApplySet parent Secret
	if !isApplySetParentSecret(secret) {
		return &admissionv1.AdmissionResponse{
			Allowed: true,
		}
	}

	logger.Info("Validating ApplySet parent Secret")

	// Validate ApplySet parent Secret structure
	var allErrors []error

	// Validate labels
	if err := s.validateParentLabels(secret, request); err != nil {
		allErrors = append(allErrors, err)
	}

	// Validate annotations
	if err := s.validateParentAnnotations(secret); err != nil {
		allErrors = append(allErrors, err)
	}

	// Check for unauthorized modifications
	if request.Operation == admissionv1.Update {
		if err := s.checkUnauthorizedModification(request); err != nil {
			allErrors = append(allErrors, err)
		}
	}

	if len(allErrors) > 0 {
		errorMsg := utilerrors.NewAggregate(allErrors).Error()
		logger.Info("Validation failed", "errors", errorMsg)
		return &admissionv1.AdmissionResponse{
			Allowed: false,
			Result: &metav1.Status{
				Message: errorMsg,
				Code:    400,
			},
		}
	}

	return &admissionv1.AdmissionResponse{
		Allowed: true,
	}
}

// mutateObject mutates an object to add default ApplySet labels if needed
func (s *WebhookServer) mutateObject(request *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
	logger := s.logger.WithValues(
		"kind", request.Kind.Kind,
		"name", request.Name,
		"namespace", request.Namespace,
		"operation", request.Operation,
	)

	// Only mutate on create/update operations
	if request.Operation == admissionv1.Delete {
		return &admissionv1.AdmissionResponse{
			Allowed: true,
		}
	}

	// Only mutate resources that might be part of an ApplySet
	// Skip Secrets (parent objects are handled by controller)
	if request.Kind.Kind == "Secret" {
		return &admissionv1.AdmissionResponse{
			Allowed: true,
		}
	}

	// Decode the object
	obj, _, err := deserializer.Decode(request.Object.Raw, nil, nil)
	if err != nil {
		logger.V(4).Info("Failed to decode object for mutation, skipping", "error", err)
		return &admissionv1.AdmissionResponse{
			Allowed: true,
		}
	}

	// Get metadata accessor
	metaObj, ok := obj.(metav1.Object)
	if !ok {
		return &admissionv1.AdmissionResponse{
			Allowed: true,
		}
	}

	// Check if object already has ApplySet label
	labels := metaObj.GetLabels()
	if labels != nil {
		if _, hasLabel := labels[labeler.ApplysetPartOfLabel]; hasLabel {
			// Already labeled, no mutation needed
			return &admissionv1.AdmissionResponse{
				Allowed: true,
			}
		}
	}

	// Check if this is a Helm-managed resource
	if !isHelmManagedResource(metaObj) {
		return &admissionv1.AdmissionResponse{
			Allowed: true,
		}
	}

	// For now, we don't mutate - the controller will handle labeling
	// This webhook is primarily for validation
	// In the future, we could add default labeling here if needed

	return &admissionv1.AdmissionResponse{
		Allowed: true,
	}
}

// validateParentLabels validates ApplySet parent Secret labels
func (s *WebhookServer) validateParentLabels(secret *v1.Secret, request *admissionv1.AdmissionRequest) error {
	labels := secret.Labels
	if labels == nil {
		return fmt.Errorf("ApplySet parent Secret %s/%s missing labels", secret.Namespace, secret.Name)
	}

	// Check for ApplySet ID label
	applySetID, ok := labels[parent.ApplySetParentIDLabel]
	if !ok {
		return fmt.Errorf("ApplySet parent Secret %s/%s missing required label %s", secret.Namespace, secret.Name, parent.ApplySetParentIDLabel)
	}

	// Validate ApplySet ID format
	if !strings.HasPrefix(applySetID, "applyset-") {
		return fmt.Errorf("ApplySet ID %q has invalid format: must start with 'applyset-'", applySetID)
	}
	if !strings.HasSuffix(applySetID, "-v1") {
		return fmt.Errorf("ApplySet ID %q has invalid format: must end with '-v1'", applySetID)
	}

	return nil
}

// validateParentAnnotations validates ApplySet parent Secret annotations
func (s *WebhookServer) validateParentAnnotations(secret *v1.Secret) error {
	annotations := secret.Annotations
	if annotations == nil {
		return fmt.Errorf("ApplySet parent Secret %s/%s missing annotations", secret.Namespace, secret.Name)
	}

	// Check for tooling annotation
	tooling, ok := annotations[parent.ApplySetToolingAnnotation]
	if !ok {
		return fmt.Errorf("ApplySet parent Secret %s/%s missing required annotation %s", secret.Namespace, secret.Name, parent.ApplySetToolingAnnotation)
	}

	// Validate tooling format (should be <tool>/<version>)
	parts := strings.Split(tooling, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf("ApplySet tooling annotation %q has invalid format: expected '<tool>/<version>'", tooling)
	}

	// Check for GroupKinds annotation (optional but should be valid if present)
	if groupKindsStr, ok := annotations[parent.ApplySetGKsAnnotation]; ok {
		if _, err := parent.ParseGroupKinds(groupKindsStr); err != nil {
			return fmt.Errorf("ApplySet parent Secret %s/%s has invalid %s annotation: %w", secret.Namespace, secret.Name, parent.ApplySetGKsAnnotation, err)
		}
	}

	return nil
}

// checkUnauthorizedModification checks if ApplySet labels/annotations are being modified by unauthorized users
func (s *WebhookServer) checkUnauthorizedModification(request *admissionv1.AdmissionRequest) error {
	// Decode old and new objects
	oldSecret := &v1.Secret{}
	newSecret := &v1.Secret{}

	if request.OldObject.Raw != nil {
		if _, _, err := deserializer.Decode(request.OldObject.Raw, nil, oldSecret); err != nil {
			return nil // Can't validate, allow
		}
	}

	if request.Object.Raw != nil {
		if _, _, err := deserializer.Decode(request.Object.Raw, nil, newSecret); err != nil {
			return nil // Can't validate, allow
		}
	}

	// Check if user is authorized (controller service account or user-agent)
	if s.isAuthorizedUser(request) {
		return nil // Authorized, allow
	}

	// Check if ApplySet labels/annotations were modified
	oldLabels := oldSecret.Labels
	newLabels := newSecret.Labels
	oldAnnotations := oldSecret.Annotations
	newAnnotations := newSecret.Annotations

	// Check for ApplySet label changes
	if oldLabels != nil && newLabels != nil {
		oldID := oldLabels[parent.ApplySetParentIDLabel]
		newID := newLabels[parent.ApplySetParentIDLabel]
		if oldID != "" && oldID != newID {
			return fmt.Errorf("unauthorized modification: %s label cannot be changed from %q to %q. Only the %s controller can modify ApplySet metadata", parent.ApplySetParentIDLabel, oldID, newID, s.controllerUA)
		}
	}

	// Check for ApplySet annotation changes
	if oldAnnotations != nil && newAnnotations != nil {
		oldTooling := oldAnnotations[parent.ApplySetToolingAnnotation]
		newTooling := newAnnotations[parent.ApplySetToolingAnnotation]
		if oldTooling != "" && oldTooling != newTooling {
			return fmt.Errorf("unauthorized modification: %s annotation cannot be changed from %q to %q. Only the %s controller can modify ApplySet metadata", parent.ApplySetToolingAnnotation, oldTooling, newTooling, s.controllerUA)
		}

		oldGKs := oldAnnotations[parent.ApplySetGKsAnnotation]
		newGKs := newAnnotations[parent.ApplySetGKsAnnotation]
		if oldGKs != "" && oldGKs != newGKs {
			// Allow updates to GroupKinds (they can change as resources are added/removed)
			// But validate the format
			if _, err := parent.ParseGroupKinds(newGKs); err != nil {
				return fmt.Errorf("invalid %s annotation format: %w", parent.ApplySetGKsAnnotation, err)
			}
		}
	}

	return nil
}

// isAuthorizedUser checks if the request is from an authorized user (controller)
func (s *WebhookServer) isAuthorizedUser(request *admissionv1.AdmissionRequest) bool {
	// Check user info
	if request.UserInfo.Username != "" {
		// Check if it's the controller service account
		// Format: system:serviceaccount:<namespace>:<serviceaccount>
		if strings.HasPrefix(request.UserInfo.Username, "system:serviceaccount:") {
			parts := strings.Split(request.UserInfo.Username, ":")
			if len(parts) == 4 && parts[3] == s.controllerSA {
				return true
			}
		}
	}

	// Check user-agent from request (if available in extra fields)
	// Note: User-agent is typically not available in AdmissionRequest
	// We rely on service account for authorization

	return false
}

// isApplySetParentSecret checks if a Secret is an ApplySet parent Secret
func isApplySetParentSecret(secret *v1.Secret) bool {
	if secret.Labels == nil {
		return false
	}

	// Check for ApplySet ID label
	_, hasLabel := secret.Labels[parent.ApplySetParentIDLabel]
	return hasLabel
}

// isHelmManagedResource checks if a resource is managed by Helm
func isHelmManagedResource(obj metav1.Object) bool {
	labels := obj.GetLabels()
	if labels == nil {
		return false
	}

	// Check for Helm labels
	managedBy, hasManagedBy := labels[labeler.HelmManagedByLabel]
	instance, hasInstance := labels[labeler.HelmInstanceLabel]

	return hasManagedBy && managedBy == "Helm" && hasInstance && instance != ""
}

// writeResponse writes an AdmissionReview response
func (s *WebhookServer) writeResponse(w http.ResponseWriter, review *admissionv1.AdmissionReview) {
	w.Header().Set("Content-Type", "application/json")

	respBytes, err := json.Marshal(review)
	if err != nil {
		s.logger.Error(err, "Failed to marshal admission response")
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if _, err := w.Write(respBytes); err != nil {
		s.logger.Error(err, "Failed to write admission response")
	}
}

// writeAllowResponse writes an allow response (used for fail-open)
func (s *WebhookServer) writeAllowResponse(w http.ResponseWriter, r *http.Request, reason string) {
	// Try to extract UID from request if possible
	// For simplicity, we'll create a minimal allow response
	response := &admissionv1.AdmissionResponse{
		Allowed: true,
		Result: &metav1.Status{
			Message: reason,
		},
	}

	review := &admissionv1.AdmissionReview{
		Response: response,
	}

	s.writeResponse(w, review)
}

// LoadTLSConfig loads TLS configuration from certificate files
func LoadTLSConfig(certFile, keyFile string) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load TLS certificate: %w", err)
	}

	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
	}, nil
}

// GenerateSelfSignedCert generates a self-signed certificate for testing
// In production, use cert-manager or similar for certificate management
func GenerateSelfSignedCert(commonName string) (*tls.Config, error) {
	// This is a placeholder - in production, use proper certificate generation
	// For now, return an error indicating cert-manager should be used
	return nil, fmt.Errorf("self-signed certificate generation not implemented. Use cert-manager or provide certificate files")
}

// Note: For production use, integrate with cert-manager or use the cert library
// to generate proper certificates. The ValidateCABundle function can be used
// to validate certificates provided by cert-manager.

// ValidateCABundle validates a CA bundle for webhook configuration
func ValidateCABundle(caBundle []byte) error {
	if len(caBundle) == 0 {
		return fmt.Errorf("CA bundle is empty")
	}

	// Try to parse as PEM
	block, _ := pem.Decode(caBundle)
	if block == nil {
		return fmt.Errorf("CA bundle is not valid PEM")
	}

	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return fmt.Errorf("failed to parse CA certificate: %w", err)
	}

	// Validate it's a CA certificate
	if !cert.IsCA {
		return fmt.Errorf("certificate is not a CA certificate")
	}

	return nil
}
