#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.iecore = None
        self.ienetwork = None
        self.input_blob = None
        self.output_blob = None
        self.ienetwork_exec = None
        self.input_image_shape = None
        

    def load_model(self, model, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        model_xml_file = model
        model_weights_file = os.path.splitext(model_xml_file)[0]+".bin"
        self.iecore = IECore()
        ### TODO: Check for supported layers ###
        network_supported_layers = self.iecore.query_network(network=self.ienetwork, device_name="CPU")
        
        not_supported_layers = []
        for layer in self.ienetwork.layers.keys():
            if layer not in network_supported_layers:
                not_supported_layers.append(layer)
        if len(not_supported_layers)>0:
            log.debug("Not supported layers in model: ".format(not_supported_layers))
            exit(1)
        self.ienetwork_exec = self.iecore.load_network(self.ienetwork, device)
        self.input_blob = next(iter(self.ienetwork.inputs))
        self.output_blob = next(iter(self.ienetwork.outputs))
        ### TODO: Add any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self.iecore.add_extension(cpu_extension, "CPU")    
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        
        return self.iecore

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.ienetwork.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request_handle = self.net_plugin.start_async(
            request_id=request_id, inputs={self.input_blob: frame})
        return self.net_plugin
    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[0].wait(-1)
        return status
    def get_output(self, request_id, output=None):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
        return res
        