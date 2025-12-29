#!/usr/bin/env python3
# file: alice.py
# Enhanced ASS (Alice Side Scripting) Interpreter with JCSS Support
# Version: 1.0.0

import sys
import os
import re
import json
import time
import random
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable

# ============================================================================
# JCSS PARSER
# ============================================================================

class JCSSParser:
    """Parse JCSS (JavaScript-CSS) files to extract ASS sections"""
    
    def __init__(self):
        self.sections = {}
    
    def parse_jcss(self, content: str) -> Dict[str, str]:
        """Parse JCSS content into sections"""
        sections = {
            'meta': '',
            'ass': '',
            'css': '',
            'javascript': '',
            'tests': ''
        }
        
        # Define section markers
        section_markers = [
            ('META-DEFINITIONS', 'meta'),
            ('ASS-SCRIPT', 'ass'),
            ('CSS-STYLES', 'css'),
            ('JAVASCRIPT', 'javascript'),
            ('TEST-SUITE', 'tests')
        ]
        
        current_section = None
        section_content = []
        brace_depth = 0
        in_comment = False
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for section start
            if line.startswith('// SECTION:'):
                # Save previous section if any
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content)
                
                # Start new section
                section_name = line.replace('// SECTION:', '').strip()
                section_key = self.map_section_name(section_name)
                
                # Skip to next non-empty line (should be section start)
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                
                if i < len(lines):
                    section_line = lines[i].strip()
                    if section_line.endswith('{'):
                        current_section = section_key
                        section_content = []
                        i += 1
                        continue
                else:
                    current_section = None
            
            # Process line if in a section
            if current_section:
                # Handle braces to find section end
                if '{' in line:
                    brace_depth += line.count('{')
                if '}' in line:
                    brace_depth -= line.count('}')
                
                section_content.append(line)
                
                # End of section
                if brace_depth <= 0:
                    sections[current_section] = '\n'.join(section_content)
                    current_section = None
                    section_content = []
                    brace_depth = 0
            
            i += 1
        
        # Handle any remaining section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
    
    def map_section_name(self, name: str) -> str:
        """Map JCSS section name to internal key"""
        name_lower = name.lower().replace(' ', '_')
        if 'meta' in name_lower:
            return 'meta'
        elif 'ass' in name_lower or 'script' in name_lower:
            return 'ass'
        elif 'css' in name_lower or 'style' in name_lower:
            return 'css'
        elif 'javascript' in name_lower or 'js' in name_lower:
            return 'javascript'
        elif 'test' in name_lower:
            return 'tests'
        return name_lower
    
    def extract_ass_from_jcss(self, content: str) -> str:
        """Extract only ASS section from JCSS content"""
        sections = self.parse_jcss(content)
        return sections.get('ass', '')

# ============================================================================
# ASS INTERPRETER CORE
# ============================================================================

class ASSInterpreter:
    """Enhanced interpreter for Alice Style Sheets with JCSS support"""
    
    def __init__(self):
        # Core state
        self.variables = {}
        self.arrays = {}
        self.functions = {}
        self.events = {}
        self.directives = {}
        self.output = []
        self.line_number = 0
        
        # JCSS parser
        self.jcss_parser = JCSSParser()
        
        # Dashboard and widget system
        self.dashboard = None
        self.widgets = {}
        self.running = True
        
        # Execution context
        self.current_function = None
        self.function_stack = []
        self.loop_stack = []
        self.condition_stack = []
        
        # Built-in functions
        self.builtins = {
            'length': self.builtin_length,
            'split': self.builtin_split,
            'join': self.builtin_join,
            'random': self.builtin_random,
            'contains': self.builtin_contains,
            'startswith': self.builtin_startswith,
            'substring': self.builtin_substring,
            'round': self.builtin_round,
            'floor': self.builtin_floor,
            'ceil': self.builtin_ceil,
            'sqrt': self.builtin_sqrt,
            'sin': self.builtin_sin,
            'cos': self.builtin_cos,
            'pi': math.pi,
            'current_timestamp': lambda: time.time() * 1000,
            'datetime': datetime.now,
            'json_parse': json.loads,
            'json_stringify': json.dumps,
        }
        
        # Domain command handlers
        self.domain_handlers = {
            'dashboard': self.handle_dashboard,
            'widget': self.handle_widget,
            'component': self.handle_component,
            'quantum': self.handle_quantum,
            'daoist': self.handle_daoist,
            'fractal': self.handle_fractal,
            'chaos': self.handle_chaos,
            'holographic': self.handle_holographic,
            'consciousness': self.handle_consciousness,
            'calculator': self.handle_calculator,
        }
    
    # Built-in function implementations
    def builtin_length(self, obj):
        if isinstance(obj, (list, tuple, str)):
            return len(obj)
        elif isinstance(obj, dict):
            return len(obj)
        return 0
    
    def builtin_split(self, text, delimiter=','):
        if isinstance(text, str):
            return text.split(delimiter)
        return []
    
    def builtin_join(self, array, delimiter=','):
        if isinstance(array, (list, tuple)):
            return delimiter.join(str(item) for item in array)
        return str(array)
    
    def builtin_random(self, start=0, end=1):
        return random.uniform(start, end)
    
    def builtin_contains(self, container, item):
        if isinstance(container, str):
            return item in container
        elif isinstance(container, (list, tuple)):
            return item in container
        elif isinstance(container, dict):
            return item in container
        return False
    
    def builtin_startswith(self, text, prefix):
        if isinstance(text, str):
            return text.startswith(prefix)
        return False
    
    def builtin_substring(self, text, start, end=None):
        if isinstance(text, str):
            if end is None:
                return text[start:]
            return text[start:end]
        return text
    
    def builtin_round(self, number, decimals=0):
        return round(number, decimals)
    
    def builtin_floor(self, number):
        return math.floor(number)
    
    def builtin_ceil(self, number):
        return math.ceil(number)
    
    def builtin_sqrt(self, number):
        return math.sqrt(number) if number >= 0 else 0
    
    def builtin_sin(self, angle):
        return math.sin(angle)
    
    def builtin_cos(self, angle):
        return math.cos(angle)
    
    def evaluate_expression(self, expr: str) -> Any:
        """Evaluate an expression string with variable substitution"""
        if not expr or not isinstance(expr, str):
            return expr
        
        # Handle boolean literals
        if expr.lower() == 'true':
            return True
        if expr.lower() == 'false':
            return False
        if expr.lower() == 'null' or expr.lower() == 'none':
            return None
        
        # Replace variables with ${variable} syntax
        def replace_var(match):
            var_name = match.group(1)
            if var_name in self.variables:
                return str(self.variables[var_name])
            elif var_name in self.arrays:
                return str(self.arrays[var_name])
            elif var_name in self.builtins:
                val = self.builtins[var_name]
                return str(val() if callable(val) else val)
            return match.group(0)  # Keep original if variable not found
        
        # Replace ${variable} patterns
        expr = re.sub(r'\$\{([^}]+)\}', replace_var, expr)
        
        # Also replace simple variable names (without ${})
        for var_name, var_value in self.variables.items():
            # Only replace whole words to avoid partial replacements
            pattern = r'\b' + re.escape(var_name) + r'\b'
            expr = re.sub(pattern, str(var_value), expr)
        
        # Try to evaluate as Python expression (safe subset)
        try:
            # Remove any trailing/leading quotes
            expr = expr.strip()
            if (expr.startswith('"') and expr.endswith('"')) or \
               (expr.startswith("'") and expr.endswith("'")):
                return expr[1:-1]
            
            # Try numeric evaluation
            try:
                # Simple arithmetic
                if '+' in expr:
                    parts = expr.split('+')
                    left = self.evaluate_expression(parts[0].strip())
                    right = self.evaluate_expression(parts[1].strip())
                    return left + right
                elif '-' in expr and not expr.startswith('-'):
                    parts = expr.split('-', 1)
                    left = self.evaluate_expression(parts[0].strip())
                    right = self.evaluate_expression(parts[1].strip())
                    return left - right
                elif '*' in expr:
                    parts = expr.split('*', 1)
                    left = self.evaluate_expression(parts[0].strip())
                    right = self.evaluate_expression(parts[1].strip())
                    return left * right
                elif '/' in expr:
                    parts = expr.split('/', 1)
                    left = self.evaluate_expression(parts[0].strip())
                    right = self.evaluate_expression(parts[1].strip())
                    return left / right if right != 0 else 0
                
                # Simple comparisons
                if '==' in expr:
                    left, right = expr.split('==', 1)
                    return self.evaluate_expression(left.strip()) == self.evaluate_expression(right.strip())
                if '!=' in expr:
                    left, right = expr.split('!=', 1)
                    return self.evaluate_expression(left.strip()) != self.evaluate_expression(right.strip())
                if '<=' in expr:
                    left, right = expr.split('<=', 1)
                    return self.evaluate_expression(left.strip()) <= self.evaluate_expression(right.strip())
                if '>=' in expr:
                    left, right = expr.split('>=', 1)
                    return self.evaluate_expression(left.strip()) >= self.evaluate_expression(right.strip())
                if '<' in expr and '>' not in expr:  # Not part of <>
                    left, right = expr.split('<', 1)
                    return self.evaluate_expression(left.strip()) < self.evaluate_expression(right.strip())
                if '>' in expr and '<' not in expr:  # Not part of <>
                    left, right = expr.split('>', 1)
                    return self.evaluate_expression(left.strip()) > self.evaluate_expression(right.strip())
                
                # Boolean operators
                if '&&' in expr:
                    parts = expr.split('&&')
                    result = True
                    for part in parts:
                        if not self.evaluate_expression(part.strip()):
                            result = False
                            break
                    return result
                if '||' in expr:
                    parts = expr.split('||')
                    result = False
                    for part in parts:
                        if self.evaluate_expression(part.strip()):
                            result = True
                            break
                    return result
                
                # Try to convert to number
                return float(expr)
            except (ValueError, TypeError):
                # If not a number, return as string
                return expr
        
        except Exception as e:
            print(f"⚠️ Error evaluating expression '{expr}': {e}")
            return expr
    
    def parse_parameters(self, param_string: str) -> Dict[str, Any]:
        """Parse parameters from a string (key=value pairs)"""
        params = {}
        
        # Split by spaces but respect quotes
        tokens = []
        current_token = ''
        in_quotes = False
        quote_char = None
        
        for char in param_string:
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current_token += char
            elif char == ' ' and not in_quotes:
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        # Parse key=value pairs
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if '=' in token:
                # Direct key=value
                key, value = token.split('=', 1)
                params[key.strip()] = self.evaluate_expression(value.strip())
            elif i + 1 < len(tokens) and tokens[i + 1] == '=' and i + 2 < len(tokens):
                # key = value format
                key = token
                value = tokens[i + 2]
                params[key.strip()] = self.evaluate_expression(value.strip())
                i += 2
            i += 1
        
        return params
    
    def execute_line(self, line: str) -> Any:
        """Execute a single line of ASS code"""
        self.line_number += 1
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            return None
        
        # Handle @directives
        if line.startswith('@'):
            parts = line[1:].split(maxsplit=1)
            directive = parts[0]
            value = parts[1] if len(parts) > 1 else True
            self.directives[directive] = value
            print(f"📌 Directive set: @{directive} = {value}")
            return None
        
        # Handle domain commands (widget, dashboard, quantum, etc.)
        for domain in self.domain_handlers:
            if line.startswith(f'{domain} '):
                return self.domain_handlers[domain](line[len(domain):].strip())
        
        # Handle echo command
        if line.startswith('echo '):
            self.handle_echo(line[5:].strip())
            return None
        
        # Handle set command
        if line.startswith('set '):
            self.handle_set(line[4:].strip())
            return None
        
        # Handle array commands
        if line.startswith('array '):
            self.handle_array(line[6:].strip())
            return None
        
        # Handle if statement
        if line.startswith('if '):
            self.handle_if(line[3:].strip())
            return None
        
        # Handle endif
        if line == 'endif':
            self.handle_endif()
            return None
        
        # Handle for loop
        if line.startswith('for '):
            self.handle_for(line[4:].strip())
            return None
        
        # Handle end (for loops, functions, etc.)
        if line == 'end':
            self.handle_end()
            return None
        
        # Handle function definition
        if line.startswith('function '):
            self.handle_function(line[9:].strip())
            return None
        
        # Handle return statement
        if line.startswith('return '):
            return self.handle_return(line[7:].strip())
        
        # Handle try
        if line == 'try':
            self.handle_try()
            return None
        
        # Handle catch
        if line.startswith('catch'):
            self.handle_catch(line[5:].strip())
            return None
        
        # Handle sleep
        if line.startswith('sleep '):
            self.handle_sleep(line[6:].strip())
            return None
        
        # Handle event registration
        if line.startswith('on event'):
            self.handle_event(line[9:].strip())
            return None
        
        # Default: treat as expression
        return self.evaluate_expression(line)
    
    def handle_echo(self, line: str):
        """Handle echo command"""
        # Check for message= prefix
        if line.startswith('message='):
            message = line[8:].strip()
        else:
            message = line
        
        # Remove quotes if present
        if (message.startswith('"') and message.endswith('"')) or \
           (message.startswith("'") and message.endswith("'")):
            message = message[1:-1]
        
        # Replace variables
        for var_name, var_value in self.variables.items():
            placeholder = '${' + var_name + '}'
            if placeholder in message:
                message = message.replace(placeholder, str(var_value))
        
        # Also check arrays
        for array_name, array_value in self.arrays.items():
            placeholder = '${' + array_name + '}'
            if placeholder in message:
                message = message.replace(placeholder, str(array_value))
        
        self.output.append(message)
        print(f"📢 {message}")
    
    def handle_set(self, line: str):
        """Handle set command"""
        # Handle array set
        if line.startswith('array '):
            parts = line[6:].split('=', 1)
            if len(parts) == 2:
                array_name = parts[0].strip()
                array_value = self.evaluate_expression(parts[1].strip())
                self.arrays[array_name] = array_value
                print(f"📚 Set array {array_name} = {array_value}")
            return
        
        # Handle regular variable assignment
        parts = line.split('=', 1)
        if len(parts) == 2:
            var_name = parts[0].strip()
            value = self.evaluate_expression(parts[1].strip())
            self.variables[var_name] = value
            print(f"💾 Set {var_name} = {value}")
    
    def handle_array(self, line: str):
        """Handle array commands"""
        parts = line.split()
        if not parts:
            return
        
        operation = parts[0]
        
        if operation == 'create':
            if len(parts) > 1:
                array_name = parts[1]
                self.arrays[array_name] = []
                print(f"📚 Created array: {array_name}")
        
        elif operation == 'push':
            if len(parts) > 2:
                array_name = parts[1]
                if parts[2] == 'value=':
                    value_str = ' '.join(parts[3:])
                    value = self.evaluate_expression(value_str)
                else:
                    value = self.evaluate_expression(parts[2])
                
                if array_name not in self.arrays:
                    self.arrays[array_name] = []
                
                self.arrays[array_name].append(value)
                print(f"📝 Pushed to {array_name}: {value}")
        
        elif operation == 'clear':
            if len(parts) > 1:
                array_name = parts[1]
                if array_name in self.arrays:
                    self.arrays[array_name] = []
                    print(f"🧹 Cleared array: {array_name}")
    
    def handle_if(self, condition: str):
        """Handle if statement"""
        result = bool(self.evaluate_expression(condition))
        self.condition_stack.append(result)
        print(f"🔀 If condition: {condition} = {result}")
    
    def handle_endif(self):
        """Handle endif"""
        if self.condition_stack:
            self.condition_stack.pop()
    
    def handle_for(self, line: str):
        """Handle for loop (simplified)"""
        print(f"🔁 For loop: {line}")
        # Simplified implementation - just record that we're in a loop
        self.loop_stack.append(True)
    
    def handle_end(self):
        """Handle end statement"""
        if self.loop_stack:
            self.loop_stack.pop()
        # Also handle end of function if needed
        if self.function_stack:
            self.function_stack.pop()
    
    def handle_function(self, line: str):
        """Handle function definition"""
        func_name = line.split()[0]
        self.functions[func_name] = line
        self.function_stack.append(func_name)
        print(f"📝 Defining function: {func_name}")
    
    def handle_return(self, line: str):
        """Handle return statement"""
        value = self.evaluate_expression(line)
        print(f"↩️ Returning: {value}")
        return value
    
    def handle_try(self):
        """Handle try block"""
        print("🛡️ Entering try block")
    
    def handle_catch(self, line: str):
        """Handle catch block"""
        print(f"🚨 Catch: {line}")
    
    def handle_sleep(self, duration: str):
        """Handle sleep command"""
        try:
            ms = float(duration)
            time.sleep(ms / 1000)
            print(f"⏸️ Slept for {ms}ms")
        except:
            pass
    
    def handle_event(self, line: str):
        """Handle event registration"""
        print(f"🎯 Event registered: {line}")
    
    # Domain command handlers
    def handle_dashboard(self, line: str):
        """Handle dashboard commands"""
        parts = line.split(maxsplit=1)
        command = parts[0] if parts else ''
        
        if command == 'create':
            params = self.parse_parameters(parts[1] if len(parts) > 1 else '')
            self.dashboard = {
                'name': params.get('name', 'Unnamed Dashboard'),
                'type': params.get('type', 'grid'),
                'columns': int(params.get('columns', 12)),
                'rows': int(params.get('rows', 8)),
                'widgets': []
            }
            print(f"📊 Created dashboard: {self.dashboard['name']}")
        
        elif command == 'apply_theme':
            params = self.parse_parameters(parts[1] if len(parts) > 1 else '')
            theme = params.get('theme', 'default')
            print(f"🎨 Applied theme: {theme}")
        
        elif command == 'resize':
            params = self.parse_parameters(parts[1] if len(parts) > 1 else '')
            width = params.get('width', 800)
            height = params.get('height', 600)
            print(f"📐 Resized dashboard to {width}x{height}")
    
    def handle_widget(self, line: str):
        """Handle widget commands - FIXED IMPLEMENTATION"""
        parts = line.split(maxsplit=1)
        command = parts[0] if parts else ''
        
        if command == 'create':
            params = self.parse_parameters(parts[1] if len(parts) > 1 else '')
            
            widget_id = params.get('id', f'widget_{len(self.widgets)}')
            widget = {
                'id': widget_id,
                'name': params.get('name', widget_id),
                'type': params.get('type', 'generic'),
                'x': int(params.get('x', 0)),
                'y': int(params.get('y', 0)),
                'width': int(params.get('width', 2)),
                'height': int(params.get('height', 2)),
                'properties': params
            }
            
            self.widgets[widget_id] = widget
            
            if self.dashboard:
                self.dashboard['widgets'].append(widget_id)
            
            print(f"📦 Created widget: {widget['name']} ({widget_id})")
            print(f"   Type: {widget['type']}")
            print(f"   Position: ({widget['x']}, {widget['y']})")
            print(f"   Size: {widget['width']}x{widget['height']}")
            if params:
                print(f"   Properties: {params}")
        
        elif command == 'move':
            params = self.parse_parameters(parts[1] if len(parts) > 1 else '')
            widget_id = params.get('id')
            
            if widget_id and widget_id in self.widgets:
                x = int(params.get('x', self.widgets[widget_id]['x']))
                y = int(params.get('y', self.widgets[widget_id]['y']))
                
                self.widgets[widget_id]['x'] = x
                self.widgets[widget_id]['y'] = y
                
                print(f"🚚 Moved widget {widget_id} to ({x}, {y})")
        
        elif command == 'resize':
            params = self.parse_parameters(parts[1] if len(parts) > 1 else '')
            widget_id = params.get('id')
            
            if widget_id and widget_id in self.widgets:
                width = int(params.get('width', self.widgets[widget_id]['width']))
                height = int(params.get('height', self.widgets[widget_id]['height']))
                
                self.widgets[widget_id]['width'] = width
                self.widgets[widget_id]['height'] = height
                
                print(f"📏 Resized widget {widget_id} to {width}x{height}")
        
        elif command == 'remove':
            params = self.parse_parameters(parts[1] if len(parts) > 1 else '')
            widget_id = params.get('id')
            
            if widget_id and widget_id in self.widgets:
                del self.widgets[widget_id]
                
                if self.dashboard and widget_id in self.dashboard['widgets']:
                    self.dashboard['widgets'].remove(widget_id)
                
                print(f"🗑️ Removed widget: {widget_id}")
        
        elif command == 'send_message':
            params = self.parse_parameters(parts[1] if len(parts) > 1 else '')
            widget_id = params.get('id')
            message = params.get('message')
            data = params.get('data', {})
            
            print(f"📨 Sent message to widget {widget_id}: {message}")
            if data:
                print(f"   Data: {data}")
        
        else:
            print(f"⚠️ Unknown widget command: {command}")
    
    def handle_component(self, line: str):
        """Handle component commands"""
        parts = line.split(maxsplit=1)
        command = parts[0] if parts else ''
        
        if command == 'initialize':
            params = self.parse_parameters(parts[1] if len(parts) > 1 else '')
            name = params.get('name', 'Unknown Component')
            version = params.get('version', '1.0.0')
            print(f"🔧 Initialized component: {name} v{version}")
    
    def handle_quantum(self, line: str):
        """Handle quantum commands"""
        print(f"⚛️ Quantum command: {line}")
    
    def handle_daoist(self, line: str):
        """Handle daoist commands"""
        print(f"☯️ Daoist command: {line}")
    
    def handle_fractal(self, line: str):
        """Handle fractal commands"""
        print(f"🌀 Fractal command: {line}")
    
    def handle_chaos(self, line: str):
        """Handle chaos commands"""
        print(f"🦋 Chaos command: {line}")
    
    def handle_holographic(self, line: str):
        """Handle holographic commands"""
        print(f"👁️ Holographic command: {line}")
    
    def handle_consciousness(self, line: str):
        """Handle consciousness commands"""
        print(f"🧘 Consciousness command: {line}")
    
    def handle_calculator(self, line: str):
        """Handle calculator commands"""
        print(f"🧮 Calculator command: {line}")
    
    def preprocess_commands(self, lines: List[str]) -> List[str]:
        """Preprocess lines to combine multi-line commands"""
        processed = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Check if this is a multi-line command (ends with nothing but next line is indented)
            if i + 1 < len(lines) and lines[i + 1].startswith(('    ', '\t')):
                # Collect all indented lines
                command_lines = [line]
                i += 1
                
                while i < len(lines) and lines[i].startswith(('    ', '\t')):
                    command_lines.append(lines[i].strip())
                    i += 1
                
                # Combine into single command
                processed.append(' '.join(command_lines))
            else:
                processed.append(line)
                i += 1
        
        return processed
    
    def execute_ass_code(self, code: str) -> bool:
        """Execute ASS code string"""
        lines = code.split('\n')
        processed_commands = self.preprocess_commands(lines)
        
        for command in processed_commands:
            try:
                result = self.execute_line(command)
                if result is not None and not self.running:
                    break
            except Exception as e:
                print(f"❌ Error executing command: {command}")
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()
        
        return True
    
    def is_jcss_file(self, content: str) -> bool:
        """Check if content is JCSS format"""
        jcss_indicators = [
            '// SECTION:',
            'meta {',
            '@version',
            '@component',
            '// SECTION: META-DEFINITIONS',
            '// SECTION: ASS-SCRIPT',
            '// SECTION: CSS-STYLES',
            '// SECTION: JAVASCRIPT'
        ]
        
        for indicator in jcss_indicators:
            if indicator in content:
                return True
        return False
    
    def parse_jcss_file(self, file_path: str) -> bool:
        """Parse and execute JCSS file"""
        print(f"📦 Parsing JCSS file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract ASS section
            ass_code = self.jcss_parser.extract_ass_from_jcss(content)
            
            if not ass_code:
                print("⚠️ No ASS section found in JCSS file, trying to parse entire file as ASS")
                ass_code = content
            
            # Remove the outer braces if present
            ass_code = ass_code.strip()
            if ass_code.startswith('{') and ass_code.endswith('}'):
                ass_code = ass_code[1:-1].strip()
            
            print(f"✅ Extracted ASS code ({len(ass_code)} characters)")
            
            # Execute the ASS code
            return self.execute_ass_code(ass_code)
            
        except Exception as e:
            print(f"❌ Error parsing JCSS file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def execute_file(self, file_path: str) -> bool:
        """Execute ASS or JCSS file"""
        print(f"🚀 Executing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Reset interpreter state for new file
            self.__init__()
            
            # Check if it's a JCSS file
            if self.is_jcss_file(content):
                return self.parse_jcss_file(file_path)
            else:
                # Regular ASS file
                return self.execute_ass_code(content)
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return False
        except Exception as e:
            print(f"❌ Error executing file: {e}")
            import traceback
            traceback.print_exc()
            return False

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for ASS interpreter"""
    if len(sys.argv) < 2:
        print("🎭 Alice Style Sheets (ASS) Interpreter v4.0.0")
        print("=============================================")
        print("Usage: python alice.py <file.ass>")
        print("       python alice.py run <file.ass>")
        print("       python alice.py <directory>")
        print()
        print("Examples:")
        print("  python alice.py index.ass")
        print("  python alice.py run widget_example.ass")
        print("  python alice.py ./my_scripts/")
        print()
        print("Supports both .ass files and .jcss (JavaScript-CSS) files")
        return
    
    command = sys.argv[1]
    
    if command == 'run':
        if len(sys.argv) < 3:
            print("Usage: python alice.py run <file.ass>")
            return
        file_path = sys.argv[2]
    else:
        file_path = command
    
    # Check if path exists
    if not os.path.exists(file_path):
        print(f"❌ Path not found: {file_path}")
        
        # Try adding .ass extension
        if not file_path.endswith('.ass') and os.path.exists(file_path + '.ass'):
            file_path += '.ass'
        elif not file_path.endswith('.jcss') and os.path.exists(file_path + '.jcss'):
            file_path += '.jcss'
        else:
            print("❌ File not found")
            return
    
    # Check if it's a directory
    if os.path.isdir(file_path):
        print(f"📁 Directory: {file_path}")
        
        # Find ASS and JCSS files
        ass_files = []
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith('.ass') or file.endswith('.jcss'):
                    ass_files.append(os.path.join(root, file))
        
        if not ass_files:
            print("❌ No .ass or .jcss files found in directory")
            return
        
        print(f"📚 Found {len(ass_files)} files:")
        for i, f in enumerate(ass_files, 1):
            print(f"  {i}. {os.path.basename(f)}")
        
        try:
            choice = input("Select file number (or 'a' for all, 'q' to quit): ")
            
            if choice.lower() == 'q':
                return
            elif choice.lower() == 'a':
                # Run all files
                for f in ass_files:
                    print(f"\n{'='*60}")
                    print(f"Running: {f}")
                    print('='*60)
                    interpreter = ASSInterpreter()
                    interpreter.execute_file(f)
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(ass_files):
                    file_path = ass_files[idx]
                else:
                    print("❌ Invalid selection")
                    return
        except ValueError:
            print("❌ Invalid input")
            return
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    # Execute the file
    interpreter = ASSInterpreter()
    success = interpreter.execute_file(file_path)
    
    if success:
        print(f"\n🎉 Execution completed!")
        print(f"📊 Variables defined: {len(interpreter.variables)}")
        print(f"📦 Widgets created: {len(interpreter.widgets)}")
        
        if interpreter.dashboard:
            print(f"\n📋 Dashboard Summary:")
            print(f"   Name: {interpreter.dashboard['name']}")
            print(f"   Type: {interpreter.dashboard['type']}")
            print(f"   Grid: {interpreter.dashboard['columns']}x{interpreter.dashboard['rows']}")
            print(f"   Widgets: {len(interpreter.dashboard['widgets'])}")
        
        if interpreter.widgets:
            print(f"\n📋 Widgets:")
            for widget_id, widget in interpreter.widgets.items():
                print(f"   - {widget['name']} ({widget_id}):")
                print(f"     Type: {widget['type']}")
                print(f"     Position: ({widget['x']}, {widget['y']})")
                print(f"     Size: {widget['width']}x{widget['height']}")
    else:
        print(f"\n❌ Execution failed")

if __name__ == '__main__':
    main()