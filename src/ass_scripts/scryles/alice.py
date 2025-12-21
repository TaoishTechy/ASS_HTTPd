#!/usr/bin/env python3
"""
alice.py - Alice Side Scripting (ASS) Language Interpreter
Version: v1.0.0

A comprehensive interpreter for the .ass (Alice Side Scripting) language,
used for procedural generation, universe creation, and quantum simulations.
"""

import os
import sys
import re
import math
import random
import json
import time
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum

# ============================================================================
# LANGUAGE TOKENIZATION AND PARSING
# ============================================================================

class TokenType(Enum):
    """Token types for the ASS language"""
    COMMENT = "#"
    COMMAND = "COMMAND"
    VARIABLE = "VARIABLE"
    STRING = "STRING"
    NUMBER = "NUMBER"
    OPERATOR = "OPERATOR"
    DELIMITER = "DELIMITER"
    KEYWORD = "KEYWORD"
    FUNCTION = "FUNCTION"
    EOF = "EOF"

class Token:
    """Token representation"""
    def __init__(self, type: TokenType, value: Any, line: int, col: int):
        self.type = type
        self.value = value
        self.line = line
        self.col = col
    
    def __repr__(self):
        return f"Token({self.type}, '{self.value}', line={self.line}, col={self.col})"

class ALexer:
    """Lexer for the ASS language"""
    
    KEYWORDS = {
        'echo', 'set', 'array', 'create', 'push', 'remove', 'if', 'endif',
        'for', 'in', 'range', 'step', 'end', 'function', 'return', 'script',
        'run', 'while', 'break', 'continue', 'true', 'false', 'null'
    }
    
    DOMAIN_COMMANDS = {
        'cosmos', 'terrain', 'world', 'dashboard', 'camera', 'renderer',
        'noise', 'vegetation', 'structure', 'river', 'lake', 'biome',
        'water', 'realm', 'civilization', 'celestial_body', 'event',
        'portal', 'visualization', 'quantum', 'universe', 'samsara',
        'widget', 'timer', 'event', 'space', 'planet', 'star', 'galaxy'
    }
    
    OPERATORS = {
        '=', '==', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/', '%',
        '+=', '-=', '*=', '/=', '%=', '++', '--', '&&', '||', '!', '&', '|', '^'
    }
    
    DELIMITERS = {
        '(', ')', '[', ']', '{', '}', ',', ':', ';', '.'
    }
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.col = 1
        self.tokens = []
    
    def tokenize(self) -> List[Token]:
        """Tokenize the source code"""
        while self.position < len(self.source):
            char = self.source[self.position]
            
            # Skip whitespace
            if char in ' \t':
                self.advance()
                continue
            
            # Skip newlines
            if char == '\n':
                self.line += 1
                self.col = 1
                self.advance()
                continue
            
            # Comments
            if char == '#':
                self.tokenize_comment()
                continue
            
            # Strings
            if char == '"':
                self.tokenize_string()
                continue
            
            # Numbers
            if char.isdigit() or (char == '.' and self.peek().isdigit()):
                self.tokenize_number()
                continue
            
            # Identifiers and keywords
            if char.isalpha() or char == '_':
                self.tokenize_identifier()
                continue
            
            # Operators
            if self.is_operator_start(char):
                self.tokenize_operator()
                continue
            
            # Delimiters
            if char in self.DELIMITERS:
                self.tokens.append(Token(TokenType.DELIMITER, char, self.line, self.col))
                self.advance()
                continue
            
            # Variable reference
            if char == '$' and self.peek() == '{':
                self.tokenize_variable_reference()
                continue
            
            # Unknown character
            self.advance()
        
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.col))
        return self.tokens
    
    def tokenize_comment(self):
        """Tokenize a comment"""
        start_pos = self.position
        while self.position < len(self.source) and self.source[self.position] != '\n':
            self.advance()
        comment = self.source[start_pos:self.position]
        self.tokens.append(Token(TokenType.COMMENT, comment, self.line, self.col - len(comment)))
    
    def tokenize_string(self):
        """Tokenize a string literal"""
        start_pos = self.position
        self.advance()  # Skip opening quote
        
        while self.position < len(self.source) and self.source[self.position] != '"':
            if self.source[self.position] == '\\' and self.peek() == '"':
                self.advance()  # Skip escape
            self.advance()
        
        if self.position >= len(self.source):
            raise SyntaxError(f"Unterminated string at line {self.line}")
        
        string_value = self.source[start_pos + 1:self.position]
        self.tokens.append(Token(TokenType.STRING, string_value, self.line, self.col - len(string_value)))
        self.advance()  # Skip closing quote
    
    def tokenize_number(self):
        """Tokenize a number"""
        start_pos = self.position
        
        # Integer part
        while self.position < len(self.source) and self.source[self.position].isdigit():
            self.advance()
        
        # Decimal part
        if self.position < len(self.source) and self.source[self.position] == '.':
            self.advance()
            while self.position < len(self.source) and self.source[self.position].isdigit():
                self.advance()
        
        # Scientific notation
        if self.position < len(self.source) and self.source[self.position].lower() == 'e':
            self.advance()
            if self.position < len(self.source) and self.source[self.position] in '+-':
                self.advance()
            while self.position < len(self.source) and self.source[self.position].isdigit():
                self.advance()
        
        number_str = self.source[start_pos:self.position]
        self.tokens.append(Token(TokenType.NUMBER, number_str, self.line, self.col - len(number_str)))
    
    def tokenize_identifier(self):
        """Tokenize an identifier or keyword"""
        start_pos = self.position
        
        while self.position < len(self.source) and (self.source[self.position].isalnum() or self.source[self.position] == '_'):
            self.advance()
        
        identifier = self.source[start_pos:self.position]
        
        # Check if it's a keyword
        if identifier in self.KEYWORDS:
            token_type = TokenType.KEYWORD
        elif identifier in self.DOMAIN_COMMANDS:
            token_type = TokenType.COMMAND
        else:
            token_type = TokenType.VARIABLE
        
        self.tokens.append(Token(token_type, identifier, self.line, self.col - len(identifier)))
    
    def tokenize_operator(self):
        """Tokenize an operator"""
        start_pos = self.position
        
        # Check for multi-character operators
        if self.position + 1 < len(self.source):
            two_char = self.source[self.position:self.position + 2]
            if two_char in self.OPERATORS:
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.OPERATOR, two_char, self.line, self.col - 2))
                return
        
        # Single character operator
        char = self.source[self.position]
        if char in self.OPERATORS:
            self.tokens.append(Token(TokenType.OPERATOR, char, self.line, self.col - 1))
            self.advance()
    
    def tokenize_variable_reference(self):
        """Tokenize a variable reference ${var}"""
        start_pos = self.position
        self.advance()  # Skip $
        self.advance()  # Skip {
        
        # Collect variable name
        var_name = ''
        while self.position < len(self.source) and self.source[self.position] != '}':
            var_name += self.source[self.position]
            self.advance()
        
        if self.position >= len(self.source):
            raise SyntaxError(f"Unterminated variable reference at line {self.line}")
        
        self.advance()  # Skip }
        
        self.tokens.append(Token(TokenType.VARIABLE, var_name, self.line, self.col - len(var_name) - 3))
    
    def is_operator_start(self, char: str) -> bool:
        """Check if character can start an operator"""
        for op in self.OPERATORS:
            if op.startswith(char):
                return True
        return False
    
    def peek(self, offset: int = 1) -> str:
        """Peek ahead in the source"""
        pos = self.position + offset
        return self.source[pos] if pos < len(self.source) else '\0'
    
    def advance(self):
        """Advance to the next character"""
        self.position += 1
        self.col += 1

# ============================================================================
# ABSTRACT SYNTAX TREE (AST)
# ============================================================================

class ASTNode:
    """Base class for AST nodes"""
    def __init__(self, token: Token = None):
        self.token = token
        self.children = []
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class ProgramNode(ASTNode):
    """Program node - root of the AST"""
    def __init__(self, statements: List[ASTNode]):
        super().__init__()
        self.statements = statements

class EchoNode(ASTNode):
    """Echo statement node"""
    def __init__(self, token: Token, message: ASTNode):
        super().__init__(token)
        self.message = message

class SetNode(ASTNode):
    """Variable assignment node"""
    def __init__(self, token: Token, var_name: str, value: ASTNode):
        super().__init__(token)
        self.var_name = var_name
        self.value = value

class ArrayCreateNode(ASTNode):
    """Array creation node"""
    def __init__(self, token: Token, array_name: str):
        super().__init__(token)
        self.array_name = array_name

class ArrayPushNode(ASTNode):
    """Array push node"""
    def __init__(self, token: Token, array_name: str, value: ASTNode):
        super().__init__(token)
        self.array_name = array_name
        self.value = value

class ArrayRemoveNode(ASTNode):
    """Array remove node"""
    def __init__(self, token: Token, array_name: str, pattern: str = ""):
        super().__init__(token)
        self.array_name = array_name
        self.pattern = pattern

class ForLoopNode(ASTNode):
    """For loop node"""
    def __init__(self, token: Token, var_name: str, start: ASTNode, end: ASTNode, 
                 step: ASTNode, body: List[ASTNode]):
        super().__init__(token)
        self.var_name = var_name
        self.start = start
        self.end = end
        self.step = step
        self.body = body

class IfNode(ASTNode):
    """If statement node"""
    def __init__(self, token: Token, condition: ASTNode, 
                 true_branch: List[ASTNode], false_branch: List[ASTNode] = None):
        super().__init__(token)
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch or []

class FunctionNode(ASTNode):
    """Function definition node"""
    def __init__(self, token: Token, name: str, params: List[str], body: List[ASTNode]):
        super().__init__(token)
        self.name = name
        self.params = params
        self.body = body

class ReturnNode(ASTNode):
    """Return statement node"""
    def __init__(self, token: Token, value: ASTNode = None):
        super().__init__(token)
        self.value = value

class ScriptCallNode(ASTNode):
    """Script call node"""
    def __init__(self, token: Token, script_name: str, params: Dict[str, ASTNode] = None):
        super().__init__(token)
        self.script_name = script_name
        self.params = params or {}

class DomainCommandNode(ASTNode):
    """Domain-specific command node"""
    def __init__(self, token: Token, command: str, args: List[ASTNode]):
        super().__init__(token)
        self.command = command
        self.args = args

class VariableNode(ASTNode):
    """Variable reference node"""
    def __init__(self, token: Token, name: str):
        super().__init__(token)
        self.name = name

class LiteralNode(ASTNode):
    """Literal value node"""
    def __init__(self, token: Token, value: Any):
        super().__init__(token)
        self.value = value

class BinaryOpNode(ASTNode):
    """Binary operation node"""
    def __init__(self, token: Token, left: ASTNode, op: str, right: ASTNode):
        super().__init__(token)
        self.left = left
        self.op = op
        self.right = right

class FunctionCallNode(ASTNode):
    """Function call node"""
    def __init__(self, token: Token, name: str, args: List[ASTNode]):
        super().__init__(token)
        self.name = name
        self.args = args

# ============================================================================
# PARSER
# ============================================================================

class AParser:
    """Parser for the ASS language"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if tokens else Token(TokenType.EOF, '', 1, 1)
    
    def parse(self) -> ProgramNode:
        """Parse the token stream into an AST"""
        statements = []
        
        while self.current_token.type != TokenType.EOF:
            if self.current_token.type == TokenType.COMMENT:
                self.advance()
                continue
            
            statement = self.parse_statement()
            if statement:
                statements.append(statement)
        
        return ProgramNode(statements)
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement"""
        token = self.current_token
        
        if token.type == TokenType.KEYWORD:
            if token.value == 'echo':
                return self.parse_echo()
            elif token.value == 'set':
                return self.parse_set()
            elif token.value == 'array':
                return self.parse_array_statement()
            elif token.value == 'for':
                return self.parse_for_loop()
            elif token.value == 'if':
                return self.parse_if_statement()
            elif token.value == 'function':
                return self.parse_function()
            elif token.value == 'return':
                return self.parse_return()
            elif token.value == 'script':
                return self.parse_script_call()
        
        elif token.type == TokenType.COMMAND:
            return self.parse_domain_command()
        
        # Unexpected token
        self.advance()
        return None
    
    def parse_echo(self) -> EchoNode:
        """Parse echo statement"""
        token = self.current_token
        self.advance()  # Skip 'echo'
        
        # Expect message="..."
        if self.current_token.type != TokenType.VARIABLE or self.current_token.value != 'message':
            raise SyntaxError(f"Expected 'message=' after echo at line {token.line}")
        
        self.advance()  # Skip 'message'
        
        if self.current_token.type != TokenType.OPERATOR or self.current_token.value != '=':
            raise SyntaxError(f"Expected '=' after 'message' at line {token.line}")
        
        self.advance()  # Skip '='
        
        # Parse the message expression
        message = self.parse_expression()
        
        return EchoNode(token, message)
    
    def parse_set(self) -> SetNode:
        """Parse set statement"""
        token = self.current_token
        self.advance()  # Skip 'set'
        
        # Parse variable name
        if self.current_token.type != TokenType.VARIABLE:
            raise SyntaxError(f"Expected variable name after 'set' at line {token.line}")
        
        var_name = self.current_token.value
        self.advance()  # Skip variable name
        
        # Parse '='
        if self.current_token.type != TokenType.OPERATOR or self.current_token.value != '=':
            raise SyntaxError(f"Expected '=' after variable name at line {token.line}")
        
        self.advance()  # Skip '='
        
        # Parse value expression
        value = self.parse_expression()
        
        return SetNode(token, var_name, value)
    
    def parse_array_statement(self) -> ASTNode:
        """Parse array-related statements"""
        token = self.current_token
        self.advance()  # Skip 'array'
        
        if self.current_token.type != TokenType.KEYWORD:
            raise SyntaxError(f"Expected array operation after 'array' at line {token.line}")
        
        operation = self.current_token.value
        self.advance()  # Skip operation
        
        if operation == 'create':
            # Parse array name
            if self.current_token.type != TokenType.VARIABLE:
                raise SyntaxError(f"Expected array name after 'array create' at line {token.line}")
            
            array_name = self.current_token.value
            self.advance()
            
            return ArrayCreateNode(token, array_name)
        
        elif operation == 'push':
            # Parse array name
            if self.current_token.type != TokenType.VARIABLE:
                raise SyntaxError(f"Expected array name after 'array push' at line {token.line}")
            
            array_name = self.current_token.value
            self.advance()
            
            # Parse 'value='
            if self.current_token.type != TokenType.VARIABLE or self.current_token.value != 'value':
                raise SyntaxError(f"Expected 'value=' after array name at line {token.line}")
            
            self.advance()  # Skip 'value'
            
            if self.current_token.type != TokenType.OPERATOR or self.current_token.value != '=':
                raise SyntaxError(f"Expected '=' after 'value' at line {token.line}")
            
            self.advance()  # Skip '='
            
            # Parse value
            value = self.parse_expression()
            
            return ArrayPushNode(token, array_name, value)
        
        elif operation == 'remove':
            # Parse array name
            if self.current_token.type != TokenType.VARIABLE:
                raise SyntaxError(f"Expected array name after 'array remove' at line {token.line}")
            
            array_name = self.current_token.value
            self.advance()
            
            # Parse optional pattern
            pattern = ""
            if self.current_token.type == TokenType.STRING:
                pattern = self.current_token.value
                self.advance()
            
            return ArrayRemoveNode(token, array_name, pattern)
        
        else:
            raise SyntaxError(f"Unknown array operation '{operation}' at line {token.line}")
    
    def parse_for_loop(self) -> ForLoopNode:
        """Parse for loop statement"""
        token = self.current_token
        self.advance()  # Skip 'for'
        
        # Parse variable name
        if self.current_token.type != TokenType.VARIABLE:
            raise SyntaxError(f"Expected variable name after 'for' at line {token.line}")
        
        var_name = self.current_token.value
        self.advance()  # Skip variable name
        
        # Parse 'in range'
        if self.current_token.type != TokenType.KEYWORD or self.current_token.value != 'in':
            raise SyntaxError(f"Expected 'in' after variable name at line {token.line}")
        
        self.advance()  # Skip 'in'
        
        if self.current_token.type != TokenType.KEYWORD or self.current_token.value != 'range':
            raise SyntaxError(f"Expected 'range' after 'in' at line {token.line}")
        
        self.advance()  # Skip 'range'
        
        # Parse start value
        start = self.parse_expression()
        
        # Parse end value
        end = self.parse_expression()
        
        # Parse optional step
        step = None
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == 'step':
            self.advance()  # Skip 'step'
            step = self.parse_expression()
        
        # Default step is 1
        if step is None:
            step = LiteralNode(Token(TokenType.NUMBER, "1", token.line, token.col), 1)
        
        # Parse body
        body = self.parse_block('end')
        
        # Skip 'end'
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == 'end':
            self.advance()
        
        return ForLoopNode(token, var_name, start, end, step, body)
    
    def parse_if_statement(self) -> IfNode:
        """Parse if statement"""
        token = self.current_token
        self.advance()  # Skip 'if'
        
        # Parse condition
        condition = self.parse_expression()
        
        # Parse true branch
        true_branch = self.parse_block('endif')
        
        # Check for else/elif (not implemented in this version)
        false_branch = []
        
        # Skip 'endif'
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == 'endif':
            self.advance()
        
        return IfNode(token, condition, true_branch, false_branch)
    
    def parse_function(self) -> FunctionNode:
        """Parse function definition"""
        token = self.current_token
        self.advance()  # Skip 'function'
        
        # Parse function name
        if self.current_token.type != TokenType.VARIABLE:
            raise SyntaxError(f"Expected function name after 'function' at line {token.line}")
        
        name = self.current_token.value
        self.advance()  # Skip function name
        
        # Parse parameters (simplified - assume no parameters for now)
        params = []
        
        # Parse body
        body = self.parse_block('end')
        
        # Skip 'end'
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == 'end':
            self.advance()
        
        return FunctionNode(token, name, params, body)
    
    def parse_return(self) -> ReturnNode:
        """Parse return statement"""
        token = self.current_token
        self.advance()  # Skip 'return'
        
        value = None
        if self.current_token.type != TokenType.EOF and self.current_token.type != TokenType.COMMENT:
            value = self.parse_expression()
        
        return ReturnNode(token, value)
    
    def parse_script_call(self) -> ScriptCallNode:
        """Parse script call"""
        token = self.current_token
        self.advance()  # Skip 'script'
        
        if self.current_token.type != TokenType.KEYWORD or self.current_token.value != 'run':
            raise SyntaxError(f"Expected 'run' after 'script' at line {token.line}")
        
        self.advance()  # Skip 'run'
        
        # Parse script name
        if self.current_token.type != TokenType.STRING:
            raise SyntaxError(f"Expected script name string after 'script run' at line {token.line}")
        
        script_name = self.current_token.value
        self.advance()
        
        return ScriptCallNode(token, script_name)
    
    def parse_domain_command(self) -> DomainCommandNode:
        """Parse domain-specific command"""
        token = self.current_token
        command = token.value
        self.advance()  # Skip command
        
        # Parse arguments
        args = []
        while (self.current_token.type != TokenType.EOF and 
               self.current_token.type != TokenType.COMMENT and
               not self.is_statement_end()):
            arg = self.parse_expression()
            args.append(arg)
        
        return DomainCommandNode(token, command, args)
    
    def parse_block(self, end_keyword: str) -> List[ASTNode]:
        """Parse a block of statements until end keyword"""
        statements = []
        
        while (self.current_token.type != TokenType.EOF and 
               not (self.current_token.type == TokenType.KEYWORD and 
                    self.current_token.value == end_keyword)):
            
            if self.current_token.type == TokenType.COMMENT:
                self.advance()
                continue
            
            statement = self.parse_statement()
            if statement:
                statements.append(statement)
        
        return statements
    
    def parse_expression(self) -> ASTNode:
        """Parse an expression"""
        return self.parse_logical_or()
    
    def parse_logical_or(self) -> ASTNode:
        """Parse logical OR expression"""
        node = self.parse_logical_and()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value == '||'):
            token = self.current_token
            self.advance()
            right = self.parse_logical_and()
            node = BinaryOpNode(token, node, '||', right)
        
        return node
    
    def parse_logical_and(self) -> ASTNode:
        """Parse logical AND expression"""
        node = self.parse_equality()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value == '&&'):
            token = self.current_token
            self.advance()
            right = self.parse_equality()
            node = BinaryOpNode(token, node, '&&', right)
        
        return node
    
    def parse_equality(self) -> ASTNode:
        """Parse equality expression"""
        node = self.parse_comparison()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value in ('==', '!=')):
            token = self.current_token
            op = self.current_token.value
            self.advance()
            right = self.parse_comparison()
            node = BinaryOpNode(token, node, op, right)
        
        return node
    
    def parse_comparison(self) -> ASTNode:
        """Parse comparison expression"""
        node = self.parse_addition()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value in ('<', '>', '<=', '>=')):
            token = self.current_token
            op = self.current_token.value
            self.advance()
            right = self.parse_addition()
            node = BinaryOpNode(token, node, op, right)
        
        return node
    
    def parse_addition(self) -> ASTNode:
        """Parse addition/subtraction expression"""
        node = self.parse_multiplication()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value in ('+', '-')):
            token = self.current_token
            op = self.current_token.value
            self.advance()
            right = self.parse_multiplication()
            node = BinaryOpNode(token, node, op, right)
        
        return node
    
    def parse_multiplication(self) -> ASTNode:
        """Parse multiplication/division expression"""
        node = self.parse_unary()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value in ('*', '/', '%')):
            token = self.current_token
            op = self.current_token.value
            self.advance()
            right = self.parse_unary()
            node = BinaryOpNode(token, node, op, right)
        
        return node
    
    def parse_unary(self) -> ASTNode:
        """Parse unary expression"""
        if (self.current_token.type == TokenType.OPERATOR and 
            self.current_token.value in ('+', '-', '!')):
            token = self.current_token
            op = self.current_token.value
            self.advance()
            right = self.parse_unary()
            # For simplicity, treat unary as binary with None left
            return BinaryOpNode(token, LiteralNode(Token(TokenType.NUMBER, "0", token.line, token.col), 0), op, right)
        
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        """Parse primary expression"""
        token = self.current_token
        
        if token.type == TokenType.NUMBER:
            self.advance()
            try:
                value = float(token.value) if '.' in token.value or 'e' in token.value.lower() else int(token.value)
            except ValueError:
                raise SyntaxError(f"Invalid number '{token.value}' at line {token.line}")
            return LiteralNode(token, value)
        
        elif token.type == TokenType.STRING:
            self.advance()
            return LiteralNode(token, token.value)
        
        elif token.type == TokenType.VARIABLE:
            # Check if it's a variable reference ${var}
            if token.value.startswith('{') and token.value.endswith('}'):
                var_name = token.value[1:-1]
                self.advance()
                return VariableNode(token, var_name)
            
            # Regular variable
            var_name = token.value
            self.advance()
            
            # Check if it's a function call
            if (self.current_token.type == TokenType.DELIMITER and 
                self.current_token.value == '('):
                self.advance()  # Skip '('
                args = []
                
                # Parse arguments
                while not (self.current_token.type == TokenType.DELIMITER and 
                          self.current_token.value == ')'):
                    arg = self.parse_expression()
                    args.append(arg)
                    
                    if self.current_token.type == TokenType.DELIMITER and self.current_token.value == ',':
                        self.advance()  # Skip ','
                
                self.advance()  # Skip ')'
                return FunctionCallNode(token, var_name, args)
            
            return VariableNode(token, var_name)
        
        elif token.type == TokenType.DELIMITER and token.value == '(':
            self.advance()  # Skip '('
            node = self.parse_expression()
            
            if not (self.current_token.type == TokenType.DELIMITER and self.current_token.value == ')'):
                raise SyntaxError(f"Expected ')' at line {token.line}")
            
            self.advance()  # Skip ')'
            return node
        
        else:
            raise SyntaxError(f"Unexpected token '{token.value}' at line {token.line}")
    
    def is_statement_end(self) -> bool:
        """Check if current token indicates end of statement"""
        return (self.current_token.type == TokenType.EOF or
                self.current_token.type == TokenType.COMMENT)
    
    def advance(self):
        """Advance to the next token"""
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = Token(TokenType.EOF, '', 0, 0)

# ============================================================================
# INTERPRETER
# ============================================================================

class AliceInterpreter:
    """Alice Side Scripting Language Interpreter"""
    
    def __init__(self):
        self.variables = {}
        self.arrays = {}
        self.functions = {}
        self.output = []
        self.return_value = None
        self.in_function = False
        
        # Built-in functions
        self.builtins = {
            'random': self._builtin_random,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'pow': math.pow,
            'floor': math.floor,
            'ceil': math.ceil,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'len': len,
            'time': time.time,
            'hash': hash,
            'str': str,
            'int': int,
            'float': float,
            'math_pi': math.pi,
            'math_e': math.e
        }
        
        # Domain command handlers
        self.domain_handlers = {}
        self.register_default_domain_handlers()
    
    def register_default_domain_handlers(self):
        """Register default domain command handlers"""
        self.domain_handlers['cosmos'] = self._handle_cosmos
        self.domain_handlers['terrain'] = self._handle_terrain
        self.domain_handlers['world'] = self._handle_world
        self.domain_handlers['dashboard'] = self._handle_dashboard
        self.domain_handlers['camera'] = self._handle_camera
        self.domain_handlers['quantum'] = self._handle_quantum
        self.domain_handlers['universe'] = self._handle_universe
    
    def interpret(self, ast: ProgramNode, params: Dict = None) -> Dict:
        """Interpret an AST"""
        self.variables = params.copy() if params else {}
        self.arrays = {}
        self.output = []
        self.return_value = None
        self.in_function = False
        
        try:
            for statement in ast.statements:
                self._execute_statement(statement)
            
            return {
                'success': True,
                'output': self.output,
                'variables': self.variables.copy(),
                'arrays': {k: v.copy() for k, v in self.arrays.items()},
                'functions': list(self.functions.keys()),
                'return': self.return_value
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': self.output,
                'variables': self.variables.copy(),
                'arrays': {k: v.copy() for k, v in self.arrays.items()}
            }
    
    def _execute_statement(self, node: ASTNode):
        """Execute a single statement node"""
        if isinstance(node, EchoNode):
            self._execute_echo(node)
        elif isinstance(node, SetNode):
            self._execute_set(node)
        elif isinstance(node, ArrayCreateNode):
            self._execute_array_create(node)
        elif isinstance(node, ArrayPushNode):
            self._execute_array_push(node)
        elif isinstance(node, ArrayRemoveNode):
            self._execute_array_remove(node)
        elif isinstance(node, ForLoopNode):
            self._execute_for_loop(node)
        elif isinstance(node, IfNode):
            self._execute_if(node)
        elif isinstance(node, FunctionNode):
            self._execute_function_definition(node)
        elif isinstance(node, ReturnNode):
            self._execute_return(node)
        elif isinstance(node, ScriptCallNode):
            self._execute_script_call(node)
        elif isinstance(node, DomainCommandNode):
            self._execute_domain_command(node)
        else:
            # Try to evaluate expression statements
            try:
                result = self._evaluate_expression(node)
                if result is not None:
                    self.output.append(str(result))
            except:
                pass
    
    def _execute_echo(self, node: EchoNode):
        """Execute echo statement"""
        message = self._evaluate_expression(node.message)
        self.output.append(str(message))
    
    def _execute_set(self, node: SetNode):
        """Execute set statement"""
        value = self._evaluate_expression(node.value)
        self.variables[node.var_name] = value
    
    def _execute_array_create(self, node: ArrayCreateNode):
        """Execute array create statement"""
        self.arrays[node.array_name] = []
    
    def _execute_array_push(self, node: ArrayPushNode):
        """Execute array push statement"""
        if node.array_name not in self.arrays:
            raise NameError(f"Array '{node.array_name}' not found")
        
        value = self._evaluate_expression(node.value)
        self.arrays[node.array_name].append(value)
    
    def _execute_array_remove(self, node: ArrayRemoveNode):
        """Execute array remove statement"""
        if node.array_name not in self.arrays:
            raise NameError(f"Array '{node.array_name}' not found")
        
        pattern = node.pattern
        if pattern:
            self.arrays[node.array_name] = [
                item for item in self.arrays[node.array_name] 
                if pattern not in str(item)
            ]
    
    def _execute_for_loop(self, node: ForLoopNode):
        """Execute for loop"""
        start = self._evaluate_expression(node.start)
        end = self._evaluate_expression(node.end)
        step = self._evaluate_expression(node.step)
        
        if not isinstance(start, (int, float)):
            raise TypeError(f"Start value must be a number, got {type(start).__name__}")
        if not isinstance(end, (int, float)):
            raise TypeError(f"End value must be a number, got {type(end).__name__}")
        if not isinstance(step, (int, float)):
            raise TypeError(f"Step value must be a number, got {type(step).__name__}")
        
        start = int(start)
        end = int(end)
        step = int(step)
        
        if step == 0:
            raise ValueError("Step cannot be zero")
        
        for i in range(start, end + 1, step):
            self.variables[node.var_name] = i
            for statement in node.body:
                self._execute_statement(statement)
    
    def _execute_if(self, node: IfNode):
        """Execute if statement"""
        condition = self._evaluate_expression(node.condition)
        
        # Convert to boolean
        condition_bool = bool(condition)
        
        if condition_bool:
            for statement in node.true_branch:
                self._execute_statement(statement)
        elif node.false_branch:
            for statement in node.false_branch:
                self._execute_statement(statement)
    
    def _execute_function_definition(self, node: FunctionNode):
        """Execute function definition"""
        self.functions[node.name] = {
            'params': node.params,
            'body': node.body
        }
    
    def _execute_return(self, node: ReturnNode):
        """Execute return statement"""
        if node.value:
            self.return_value = self._evaluate_expression(node.value)
        else:
            self.return_value = None
        
        # Set flag to break out of function execution
        self.in_function = False
    
    def _execute_script_call(self, node: ScriptCallNode):
        """Execute script call"""
        # In a real implementation, this would load and execute another script
        self.output.append(f"[SCRIPT] Calling script: {node.script_name}")
    
    def _execute_domain_command(self, node: DomainCommandNode):
        """Execute domain command"""
        if node.command in self.domain_handlers:
            # Evaluate arguments
            args = [self._evaluate_expression(arg) for arg in node.args]
            
            # Call handler
            result = self.domain_handlers[node.command](*args)
            if result:
                self.output.append(str(result))
        else:
            self.output.append(f"[DOMAIN] Unknown command: {node.command}")
    
    def _evaluate_expression(self, node: ASTNode) -> Any:
        """Evaluate an expression node"""
        if isinstance(node, LiteralNode):
            return node.value
        
        elif isinstance(node, VariableNode):
            # Check if variable exists
            if node.name in self.variables:
                return self.variables[node.name]
            elif node.name in self.builtins:
                return self.builtins[node.name]
            else:
                raise NameError(f"Variable '{node.name}' not found")
        
        elif isinstance(node, BinaryOpNode):
            left = self._evaluate_expression(node.left)
            right = self._evaluate_expression(node.right)
            
            # Handle operations
            if node.op == '+':
                # Handle string concatenation
                if isinstance(left, str) or isinstance(right, str):
                    return str(left) + str(right)
                return left + right
            elif node.op == '-':
                return left - right
            elif node.op == '*':
                return left * right
            elif node.op == '/':
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left / right
            elif node.op == '%':
                return left % right
            elif node.op == '==':
                return left == right
            elif node.op == '!=':
                return left != right
            elif node.op == '<':
                return left < right
            elif node.op == '>':
                return left > right
            elif node.op == '<=':
                return left <= right
            elif node.op == '>=':
                return left >= right
            elif node.op == '&&':
                return bool(left) and bool(right)
            elif node.op == '||':
                return bool(left) or bool(right)
            else:
                raise ValueError(f"Unknown operator '{node.op}'")
        
        elif isinstance(node, FunctionCallNode):
            # Handle built-in functions
            if node.name in self.builtins:
                args = [self._evaluate_expression(arg) for arg in node.args]
                return self.builtins[node.name](*args)
            
            # Handle user-defined functions
            elif node.name in self.functions:
                func = self.functions[node.name]
                
                # Evaluate arguments
                args = [self._evaluate_expression(arg) for arg in node.args]
                
                # Create local scope
                old_variables = self.variables.copy()
                
                # Set parameters
                for i, param in enumerate(func['params']):
                    if i < len(args):
                        self.variables[param] = args[i]
                
                # Execute function body
                self.in_function = True
                return_value = None
                
                try:
                    for statement in func['body']:
                        self._execute_statement(statement)
                        if not self.in_function:  # Return was called
                            return_value = self.return_value
                            break
                finally:
                    # Restore variables
                    self.variables = old_variables
                    self.in_function = False
                
                return return_value
            
            else:
                raise NameError(f"Function '{node.name}' not found")
        
        else:
            raise TypeError(f"Cannot evaluate node of type {type(node).__name__}")
    
    def _builtin_random(self, a=0, b=1):
        """Built-in random function"""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return random.randint(int(a), int(b))
        return random.random()
    
    # Domain command handlers
    def _handle_cosmos(self, *args):
        return f"[COSMOS] Creating universe with args: {args}"
    
    def _handle_terrain(self, *args):
        return f"[TERRAIN] Generating terrain with args: {args}"
    
    def _handle_world(self, *args):
        return f"[WORLD] Managing world with args: {args}"
    
    def _handle_dashboard(self, *args):
        return f"[DASHBOARD] Updating dashboard with args: {args}"
    
    def _handle_camera(self, *args):
        return f"[CAMERA] Adjusting camera with args: {args}"
    
    def _handle_quantum(self, *args):
        return f"[QUANTUM] Applying quantum effects with args: {args}"
    
    def _handle_universe(self, *args):
        return f"[UNIVERSE] Generating universe with args: {args}"

# ============================================================================
# COMPLETE ASS LANGUAGE PROCESSOR
# ============================================================================

class AliceLanguageProcessor:
    """Complete processor for the Alice Side Scripting language"""
    
    def __init__(self, script_dir: str = "./ass_scripts"):
        self.script_dir = Path(script_dir)
        self.interpreter = AliceInterpreter()
        self.script_registry = {}
        self.dependency_graph = defaultdict(list)
        
        # Load all scripts on initialization
        if self.script_dir.exists():
            self._load_all_scripts()
    
    def _load_all_scripts(self):
        """Load all .ass scripts from the script directory"""
        for root, dirs, files in os.walk(self.script_dir):
            for file in files:
                if file.endswith('.ass'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Create script key
                        rel_path = file_path.relative_to(self.script_dir)
                        script_key = str(rel_path).replace('/', '_').replace('.', '_')
                        
                        # Parse and store
                        lexer = ALexer(content)
                        tokens = lexer.tokenize()
                        parser = AParser(tokens)
                        ast = parser.parse()
                        
                        self.script_registry[script_key] = {
                            'path': str(file_path),
                            'name': file,
                            'content': content,
                            'ast': ast,
                            'size': len(content),
                            'category': root.split('/')[-1] if '/' in str(root) else 'root',
                            'loaded': datetime.now().isoformat()
                        }
                        
                        # Build dependency graph (simple detection)
                        for line in content.splitlines():
                            if 'script run' in line:
                                match = re.search(r'script run "([^"]+\.ass)"', line)
                                if match:
                                    dep_script = match.group(1).replace('.', '_')
                                    self.dependency_graph[dep_script].append(script_key)
                        
                    except Exception as e:
                        print(f"Error loading script {file_path}: {e}")
    
    def execute_script(self, script_name: str, params: Dict = None) -> Dict:
        """Execute a script by name"""
        if script_name not in self.script_registry:
            return {
                'success': False,
                'error': f"Script '{script_name}' not found",
                'available_scripts': list(self.script_registry.keys())
            }
        
        script_data = self.script_registry[script_name]
        result = self.interpreter.interpret(script_data['ast'], params)
        
        # Add script metadata
        result['script'] = script_name
        result['path'] = script_data['path']
        result['category'] = script_data['category']
        
        return result
    
    def execute_raw(self, script_content: str, params: Dict = None) -> Dict:
        """Execute raw script content"""
        try:
            lexer = ALexer(script_content)
            tokens = lexer.tokenize()
            parser = AParser(tokens)
            ast = parser.parse()
            
            result = self.interpreter.interpret(ast, params)
            result['success'] = True
            return result
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': []
            }
    
    def execute_all_in_order(self) -> Dict:
        """Execute all scripts in dependency order"""
        executed = set()
        results = {}
        
        def execute_with_deps(script_name):
            if script_name in executed:
                return
            
            # Execute dependencies first
            for dep in self.dependency_graph.get(script_name, []):
                execute_with_deps(dep)
            
            # Execute this script
            results[script_name] = self.execute_script(script_name)
            executed.add(script_name)
        
        for script_name in self.script_registry:
            execute_with_deps(script_name)
        
        return {
            'success': True,
            'total_scripts': len(results),
            'results': results
        }
    
    def register_domain_handler(self, command: str, handler: Callable):
        """Register a custom domain command handler"""
        self.interpreter.domain_handlers[command] = handler
    
    def get_script_info(self) -> Dict:
        """Get information about all loaded scripts"""
        categories = defaultdict(list)
        for script_key, script_data in self.script_registry.items():
            category = script_data.get('category', 'uncategorized')
            categories[category].append({
                'name': script_key,
                'file': script_data['name'],
                'size': script_data['size']
            })
        
        return {
            'total_scripts': len(self.script_registry),
            'scripts_by_category': dict(categories),
            'script_list': list(self.script_registry.keys()),
            'dependency_graph': dict(self.dependency_graph)
        }

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for the Alice interpreter"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Alice Side Scripting (ASS) Language Interpreter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  alice.py run script.ass                    # Run a script file
  alice.py run script.ass --param size=1024  # Run with parameters
  alice.py eval 'echo message="Hello"'       # Evaluate inline script
  alice.py repl                             # Start interactive REPL
  alice.py list                             # List loaded scripts
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a script file')
    run_parser.add_argument('script', help='Script file to run')
    run_parser.add_argument('--param', '-p', action='append', help='Parameters (key=value)')
    run_parser.add_argument('--script-dir', '-d', default='./ass_scripts', help='Script directory')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate inline script')
    eval_parser.add_argument('code', help='Script code to evaluate')
    eval_parser.add_argument('--param', '-p', action='append', help='Parameters (key=value)')
    
    # REPL command
    repl_parser = subparsers.add_parser('repl', help='Start interactive REPL')
    repl_parser.add_argument('--script-dir', '-d', default='./ass_scripts', help='Script directory')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List loaded scripts')
    list_parser.add_argument('--script-dir', '-d', default='./ass_scripts', help='Script directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Process parameters
    params = {}
    if hasattr(args, 'param') and args.param:
        for param in args.param:
            if '=' in param:
                key, value = param.split('=', 1)
                # Try to convert to number if possible
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                params[key] = value
    
    # Create processor
    script_dir = getattr(args, 'script_dir', './ass_scripts')
    processor = AliceLanguageProcessor(script_dir)
    
    # Execute command
    if args.command == 'run':
        # Read script file
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script file '{args.script}' not found")
            return 1
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # Execute
        result = processor.execute_raw(script_content, params)
        
        # Print output
        if result['success']:
            for line in result['output']:
                print(line)
        else:
            print(f"Error: {result['error']}")
            return 1
    
    elif args.command == 'eval':
        # Execute inline code
        result = processor.execute_raw(args.code, params)
        
        # Print output
        if result['success']:
            for line in result['output']:
                print(line)
        else:
            print(f"Error: {result['error']}")
            return 1
    
    elif args.command == 'repl':
        # Start interactive REPL
        print("Alice Side Scripting REPL (Type 'exit' to quit)")
        print("==============================================")
        
        while True:
            try:
                # Read input
                line = input("alice> ").strip()
                
                if line.lower() in ('exit', 'quit', 'q'):
                    break
                
                if not line:
                    continue
                
                # Execute
                result = processor.execute_raw(line)
                
                # Print output
                if result['success']:
                    for output_line in result['output']:
                        print(output_line)
                else:
                    print(f"Error: {result['error']}")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.command == 'list':
        # List loaded scripts
        info = processor.get_script_info()
        
        print(f"Loaded scripts: {info['total_scripts']}")
        print("=" * 50)
        
        for category, scripts in info['scripts_by_category'].items():
            print(f"\n{category}:")
            print("-" * 30)
            for script in scripts:
                print(f"  {script['name']} ({script['file']}) - {script['size']} bytes")
    
    return 0

# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

EXAMPLE_SCRIPTS = {
    'hello_world': '''echo message="Hello, Alice World!"''',
    
    'variables': '''set name="Alice"
set version=1.0
set pi=${math_pi}
echo message="Welcome to ${name} v${version}"
echo message="Pi is approximately ${pi}"''',
    
    'arrays': '''array create colors
array push colors value="red"
array push colors value="green"
array push colors value="blue"
echo message="Colors: ${colors}"''',
    
    'loops': '''echo message="Counting..."
for i in range 1 5
    echo message="Number: ${i}"
end
echo message="Done!"''',
    
    'math': '''set a=10
set b=20
set sum=${a} + ${b}
set product=${a} * ${b}
echo message="Sum: ${sum}, Product: ${product}"
echo message="Random number: ${random(1, 100)}"''',
    
    'domain': '''echo message="Creating terrain..."
terrain generate size=1024 seed=42
cosmos create name="Test_Universe" stars=1000
dashboard update status="ready"''',
}

def run_example(example_name: str):
    """Run an example script"""
    if example_name not in EXAMPLE_SCRIPTS:
        print(f"Example '{example_name}' not found. Available examples: {list(EXAMPLE_SCRIPTS.keys())}")
        return
    
    processor = AliceLanguageProcessor()
    result = processor.execute_raw(EXAMPLE_SCRIPTS[example_name])
    
    print(f"Running example: {example_name}")
    print("=" * 50)
    
    if result['success']:
        for line in result['output']:
            print(line)
    else:
        print(f"Error: {result['error']}")

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'ALexer',
    'AParser',
    'AliceInterpreter',
    'AliceLanguageProcessor',
    'Token',
    'TokenType',
    'run_example'
]

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())