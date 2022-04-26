//
// Copyright 2021, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package gitlab

import (
	"fmt"
	"time"
)

// BroadcastMessagesService handles communication with the broadcast
// messages methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/broadcast_messages.html
type BroadcastMessagesService struct {
	client *Client
}

// BroadcastMessage represents a GitLab issue board.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/broadcast_messages.html#get-all-broadcast-messages
type BroadcastMessage struct {
	Message  string     `json:"message"`
	StartsAt *time.Time `json:"starts_at"`
	EndsAt   *time.Time `json:"ends_at"`
	Color    string     `json:"color"`
	Font     string     `json:"font"`
	ID       int        `json:"id"`
	Active   bool       `json:"active"`
}

// ListBroadcastMessagesOptions represents the available ListBroadcastMessages()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/broadcast_messages.html#get-all-broadcast-messages
type ListBroadcastMessagesOptions ListOptions

// ListBroadcastMessages gets a list of all broadcasted messages.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/broadcast_messages.html#get-all-broadcast-messages
func (s *BroadcastMessagesService) ListBroadcastMessages(opt *ListBroadcastMessagesOptions, options ...RequestOptionFunc) ([]*BroadcastMessage, *Response, error) {
	req, err := s.client.NewRequest("GET", "broadcast_messages", opt, options)
	if err != nil {
		return nil, nil, err
	}

	var bs []*BroadcastMessage
	resp, err := s.client.Do(req, &bs)
	if err != nil {
		return nil, resp, err
	}

	return bs, resp, err
}

// GetBroadcastMessage gets a single broadcast message.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/broadcast_messages.html#get-a-specific-broadcast-message
func (s *BroadcastMessagesService) GetBroadcastMessage(broadcast int, options ...RequestOptionFunc) (*BroadcastMessage, *Response, error) {
	u := fmt.Sprintf("broadcast_messages/%d", broadcast)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	b := new(BroadcastMessage)
	resp, err := s.client.Do(req, &b)
	if err != nil {
		return nil, resp, err
	}

	return b, resp, err
}

// CreateBroadcastMessageOptions represents the available CreateBroadcastMessage()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/broadcast_messages.html#create-a-broadcast-message
type CreateBroadcastMessageOptions struct {
	Message  *string    `url:"message" json:"message"`
	StartsAt *time.Time `url:"starts_at,omitempty" json:"starts_at,omitempty"`
	EndsAt   *time.Time `url:"ends_at,omitempty" json:"ends_at,omitempty"`
	Color    *string    `url:"color,omitempty" json:"color,omitempty"`
	Font     *string    `url:"font,omitempty" json:"font,omitempty"`
}

// CreateBroadcastMessage creates a message to broadcast.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/broadcast_messages.html#create-a-broadcast-message
func (s *BroadcastMessagesService) CreateBroadcastMessage(opt *CreateBroadcastMessageOptions, options ...RequestOptionFunc) (*BroadcastMessage, *Response, error) {
	req, err := s.client.NewRequest("POST", "broadcast_messages", opt, options)
	if err != nil {
		return nil, nil, err
	}

	b := new(BroadcastMessage)
	resp, err := s.client.Do(req, &b)
	if err != nil {
		return nil, resp, err
	}

	return b, resp, err
}

// UpdateBroadcastMessageOptions represents the available CreateBroadcastMessage()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/broadcast_messages.html#update-a-broadcast-message
type UpdateBroadcastMessageOptions struct {
	Message  *string    `url:"message,omitempty" json:"message,omitempty"`
	StartsAt *time.Time `url:"starts_at,omitempty" json:"starts_at,omitempty"`
	EndsAt   *time.Time `url:"ends_at,omitempty" json:"ends_at,omitempty"`
	Color    *string    `url:"color,omitempty" json:"color,omitempty"`
	Font     *string    `url:"font,omitempty" json:"font,omitempty"`
}

// UpdateBroadcastMessage update a broadcasted message.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/broadcast_messages.html#update-a-broadcast-message
func (s *BroadcastMessagesService) UpdateBroadcastMessage(broadcast int, opt *UpdateBroadcastMessageOptions, options ...RequestOptionFunc) (*BroadcastMessage, *Response, error) {
	u := fmt.Sprintf("broadcast_messages/%d", broadcast)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	b := new(BroadcastMessage)
	resp, err := s.client.Do(req, &b)
	if err != nil {
		return nil, resp, err
	}

	return b, resp, err
}

// DeleteBroadcastMessage deletes a broadcasted message.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/broadcast_messages.html#delete-a-broadcast-message
func (s *BroadcastMessagesService) DeleteBroadcastMessage(broadcast int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("broadcast_messages/%d", broadcast)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
