/*
Copyright 2014 Google Inc. All rights reserved.

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

package registrystorage

import (
	"errors"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/RangelReale/osin"

	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	oauthclient "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/client"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/scope"
)

// UserConversion populates and extracts UserData in access and authorize tokens
type UserConversion interface {
	// ConvertToAuthorizeToken populates the UserData field in the given AuthorizeToken
	ConvertToAuthorizeToken(interface{}, *oapi.OAuthAuthorizeToken) error
	// ConvertToAccessToken populates the UserData field in the given AccessToken
	ConvertToAccessToken(interface{}, *oapi.OAuthAccessToken) error
	// ConvertFromAuthorizeToken extracts user data from the given AuthorizeToken
	ConvertFromAuthorizeToken(*oapi.OAuthAuthorizeToken) (interface{}, error)
	// ConvertFromAccessToken extracts user data from the given AccessToken
	ConvertFromAccessToken(*oapi.OAuthAccessToken) (interface{}, error)
}

// storage implements osin.Storage backed by API registries
type storage struct {
	client oauthclient.Interface
	user   UserConversion
}

// New returns an osin.Storage object backed by the given registries
func New(client oauthclient.Interface, user UserConversion) osin.Storage {
	return &storage{
		client: client,
		user:   user,
	}
}

// Clone the storage if needed. For example, using mgo, you can clone the session with session.Clone
// to avoid concurrent access problems.
// This is to avoid cloning the connection at each method access.
// Can return itself if not a problem.
func (s *storage) Clone() osin.Storage {
	return s
}

// Close the resources the Storage potentially holds (using Clone for example)
func (s *storage) Close() {
}

// GetClient loads the client by id (client_id)
func (s *storage) GetClient(id string) (osin.Client, error) {
	c, err := s.client.OAuthClients().Get(id)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return &clientWrapper{id, c}, nil
}

// SaveAuthorize saves authorize data.
func (s *storage) SaveAuthorize(data *osin.AuthorizeData) error {
	token, err := s.convertToAuthorizeToken(data)
	if err != nil {
		return err
	}
	_, err = s.client.OAuthAuthorizeTokens().Create(token)
	return err
}

// LoadAuthorize looks up AuthorizeData by a code.
// Client information MUST be loaded together.
// Optionally can return error if expired.
func (s *storage) LoadAuthorize(code string) (*osin.AuthorizeData, error) {
	authorize, err := s.client.OAuthAuthorizeTokens().Get(code)
	if err != nil {
		return nil, err
	}
	user, err := s.user.ConvertFromAuthorizeToken(authorize)
	if err != nil {
		return nil, err
	}
	client, err := s.client.OAuthClients().Get(authorize.ClientName)
	if err != nil {
		return nil, err
	}

	return &osin.AuthorizeData{
		Code:        code,
		Client:      &clientWrapper{authorize.ClientName, client},
		ExpiresIn:   int32(authorize.ExpiresIn),
		Scope:       scope.Join(authorize.Scopes),
		RedirectUri: authorize.RedirectURI,
		State:       authorize.State,
		CreatedAt:   authorize.CreationTimestamp.Time,
		UserData:    user,
	}, nil
}

// RemoveAuthorize revokes or deletes the authorization code.
func (s *storage) RemoveAuthorize(code string) error {
	err := s.client.OAuthAuthorizeTokens().Delete(code)
	if err != nil && apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

// SaveAccess writes AccessData.
// If RefreshToken is not blank, it must save in a way that can be loaded using LoadRefresh.
func (s *storage) SaveAccess(data *osin.AccessData) error {
	token, err := s.convertToAccessToken(data)
	if err != nil {
		return err
	}
	_, err = s.client.OAuthAccessTokens().Create(token)
	return err
}

// LoadAccess retrieves access data by token. Client information MUST be loaded together.
// AuthorizeData and AccessData DON'T NEED to be loaded if not easily available.
// Optionally can return error if expired.
func (s *storage) LoadAccess(token string) (*osin.AccessData, error) {
	access, err := s.client.OAuthAccessTokens().Get(token)
	if err != nil {
		return nil, err
	}
	user, err := s.user.ConvertFromAccessToken(access)
	if err != nil {
		return nil, err
	}
	client, err := s.client.OAuthClients().Get(access.ClientName)
	if err != nil {
		return nil, err
	}

	return &osin.AccessData{
		AccessToken:  token,
		RefreshToken: access.RefreshToken,
		Client:       &clientWrapper{access.ClientName, client},
		ExpiresIn:    int32(access.ExpiresIn),
		Scope:        scope.Join(access.Scopes),
		RedirectUri:  access.RedirectURI,
		CreatedAt:    access.CreationTimestamp.Time,
		UserData:     user,
	}, nil
}

// RemoveAccess revokes or deletes an AccessData.
func (s *storage) RemoveAccess(token string) error {
	err := s.client.OAuthAccessTokens().Delete(token)
	if err != nil && apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

// LoadRefresh retrieves refresh AccessData. Client information MUST be loaded together.
// AuthorizeData and AccessData DON'T NEED to be loaded if not easily available.
// Optionally can return error if expired.
func (s *storage) LoadRefresh(token string) (*osin.AccessData, error) {
	// TODO: lookup by refresh token
	return nil, errors.New("not implemented")
}

// RemoveRefresh revokes or deletes refresh AccessData.
func (s *storage) RemoveRefresh(token string) error {
	// TODO: delete by refresh token
	return errors.New("not implemented")
}

// clientWrapper implements the osin.Client interface by wrapping an API Client object
type clientWrapper struct {
	id     string
	client *oapi.OAuthClient
}

// GetId implements osin.Client
func (w *clientWrapper) GetId() string {
	return w.id
}

// GetSecret implements osin.Client
func (w *clientWrapper) GetSecret() string {
	return w.client.Secret
}

// GetRedirectUri implements osin.Client
func (w *clientWrapper) GetRedirectUri() string {
	if len(w.client.RedirectURIs) == 0 {
		return ""
	}
	return strings.Join(w.client.RedirectURIs, ",")
}

// GetUserData implements osin.Client
func (w *clientWrapper) GetUserData() interface{} {
	return nil
}

func (s *storage) convertToAuthorizeToken(data *osin.AuthorizeData) (*oapi.OAuthAuthorizeToken, error) {
	token := &oapi.OAuthAuthorizeToken{
		ObjectMeta: api.ObjectMeta{
			Name:              data.Code,
			CreationTimestamp: util.Time{data.CreatedAt},
		},
		ClientName:  data.Client.GetId(),
		ExpiresIn:   int64(data.ExpiresIn),
		Scopes:      scope.Split(data.Scope),
		RedirectURI: data.RedirectUri,
		State:       data.State,
	}
	if err := s.user.ConvertToAuthorizeToken(data.UserData, token); err != nil {
		return nil, err
	}
	return token, nil
}

func (s *storage) convertToAccessToken(data *osin.AccessData) (*oapi.OAuthAccessToken, error) {
	token := &oapi.OAuthAccessToken{
		ObjectMeta: api.ObjectMeta{
			Name:              data.AccessToken,
			CreationTimestamp: util.Time{data.CreatedAt},
		},
		ExpiresIn:    int64(data.ExpiresIn),
		RefreshToken: data.RefreshToken,
		ClientName:   data.Client.GetId(),
		Scopes:       scope.Split(data.Scope),
		RedirectURI:  data.RedirectUri,
	}
	if data.AuthorizeData != nil {
		authToken, err := s.convertToAuthorizeToken(data.AuthorizeData)
		if err != nil {
			return nil, err
		}
		token.AuthorizeToken = *authToken
	}
	if err := s.user.ConvertToAccessToken(data.UserData, token); err != nil {
		return nil, err
	}
	return token, nil
}
