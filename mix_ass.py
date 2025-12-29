#!/usr/bin/env python3
"""
mix_ass.py - God-Tier Multi-Language Pattern-Driven Extractor & Converter
Uses external JSON pattern files for syntax recognition and conversion
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import html
import traceback
from dataclasses import dataclass, field
from enum import Enum
import yaml

class LanguageType(Enum):
    PHP = "php"
    HTML = "html"
    JS = "javascript"
    CSS = "css"
    JCSS = "jcss"
    ASS = "ass"
    PYTHON = "python"
    SQL = "sql"
    UNKNOWN = "unknown"

@dataclass
class SyntaxPattern:
    """Represents a syntax pattern for language detection"""
    name: str
    regex: str
    capture_groups: List[str] = field(default_factory=list)
    flags: int = 0
    weight: float = 1.0  # Confidence weight for detection
    
@dataclass
class LanguagePattern:
    """Complete language pattern definition"""
    language: str
    extensions: List[str]
    syntax_patterns: Dict[str, Any]
    flow_patterns: Dict[str, Any]
    conversion_patterns: Dict[str, Any]
    context_rules: Dict[str, Any]
    
class PatternRegistry:
    """Registry for loading and managing language patterns"""
    
    def __init__(self, pattern_dir: str = "patterns"):
        self.pattern_dir = Path(pattern_dir)
        self.patterns: Dict[str, LanguagePattern] = {}
        self.load_patterns()
    
    def load_patterns(self):
        """Load all JSON pattern files from directory"""
        if not self.pattern_dir.exists():
            # Create default patterns directory
            self.pattern_dir.mkdir(exist_ok=True)
            self._create_default_patterns()
        
        for pattern_file in self.pattern_dir.glob("*.json"):
            try:
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    pattern_data = json.load(f)
                
                language = pattern_data.get('language')
                if language:
                    self.patterns[language] = LanguagePattern(
                        language=language,
                        extensions=pattern_data.get('extensions', []),
                        syntax_patterns=pattern_data.get('syntax_patterns', {}),
                        flow_patterns=pattern_data.get('flow_patterns', {}),
                        conversion_patterns=pattern_data.get('conversion_patterns', {}),
                        context_rules=pattern_data.get('context_rules', {})
                    )
                    print(f"✅ Loaded pattern for: {language}")
                    
            except Exception as e:
                print(f"❌ Error loading pattern {pattern_file}: {e}")
    
    def _create_default_patterns(self):
        """Create default pattern files"""
        default_patterns = {
            'php_syntax.json': {
                "language": "php",
                "extensions": [".php", ".phtml", ".inc"],
                "syntax_patterns": {
                    "php_tags": {"open": ["<?php", "<?="], "close": ["?>"]},
                    "variables": {"regex": "\\$(\\w+)\\s*="},
                    "functions": {"regex": "function\\s+(\\w+)\\s*\\("},
                    "classes": {"regex": "class\\s+(\\w+)"}
                }
            },
            'html_syntax.json': {
                "language": "html",
                "extensions": [".html", ".htm"],
                "syntax_patterns": {
                    "elements": {"regex": "<([a-zA-Z][^>]*)>"},
                    "attributes": {"regex": "([a-zA-Z-]+)=\"[^\"]*\""}
                }
            },
            'js_syntax.json': {
                "language": "javascript",
                "extensions": [".js", ".jsx"],
                "syntax_patterns": {
                    "variables": {"regex": "(var|let|const)\\s+(\\w+)\\s*="},
                    "functions": {"regex": "function\\s+(\\w+)\\s*\\("}
                }
            }
        }
        
        for filename, content in default_patterns.items():
            with open(self.pattern_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2)
    
    def detect_language(self, content: str, filename: str = "") -> Tuple[str, float]:
        """Detect language with confidence score"""
        ext = Path(filename).suffix.lower() if filename else ""
        scores = {}
        
        for lang_name, pattern in self.patterns.items():
            score = 0.0
            
            # Check extension match
            if ext in pattern.extensions:
                score += 0.3
            
            # Check for language-specific patterns
            for category, patterns in pattern.syntax_patterns.items():
                if isinstance(patterns, dict) and 'regex' in patterns:
                    if re.search(patterns['regex'], content, re.DOTALL):
                        score += 0.2
                elif isinstance(patterns, list):
                    for p in patterns:
                        if isinstance(p, dict) and 'regex' in p:
                            if re.search(p['regex'], content, re.DOTALL):
                                score += 0.1
            
            # Check for flow patterns
            for flow_name, flow_pattern in pattern.flow_patterns.items():
                if isinstance(flow_pattern, dict) and 'pattern' in flow_pattern:
                    if re.search(flow_pattern['pattern'], content, re.DOTALL):
                        score += 0.15
            
            if score > 0:
                scores[lang_name] = score
        
        if scores:
            best_lang = max(scores.items(), key=lambda x: x[1])
            return best_lang[0], best_lang[1]
        
        # Fallback detection
        if '<?php' in content or '$_' in content:
            return 'php', 0.8
        elif '<!DOCTYPE' in content or '<html' in content:
            return 'html', 0.8
        elif 'function' in content and ('var ' in content or 'let ' in content):
            return 'javascript', 0.8
        elif '$' in content and ':' in content and ';' in content:
            return 'jcss', 0.7
        elif 'ass {' in content or 'class ' in content:
            return 'ass', 0.7
        
        return 'unknown', 0.0
    
    def get_pattern(self, language: str) -> Optional[LanguagePattern]:
        """Get pattern for specific language"""
        return self.patterns.get(language)

class PatternDrivenParser:
    """Parser that uses loaded patterns to extract syntax"""
    
    def __init__(self, pattern_registry: PatternRegistry):
        self.registry = pattern_registry
    
    def parse_content(self, content: str, language: str = "") -> Dict[str, Any]:
        """Parse content using language patterns"""
        if not language:
            language, confidence = self.registry.detect_language(content)
        
        pattern = self.registry.get_pattern(language)
        if not pattern:
            # Try to parse with generic patterns
            return self._parse_generic(content, language)
        
        result = {
            'language': language,
            'confidence': confidence if 'confidence' in locals() else 1.0,
            'raw': content,
            'elements': {},
            'flow': [],
            'contexts': []
        }
        
        # Parse syntax patterns
        result['elements'] = self._parse_syntax_patterns(content, pattern.syntax_patterns)
        
        # Parse flow patterns
        result['flow'] = self._parse_flow_patterns(content, pattern.flow_patterns)
        
        # Detect contexts
        result['contexts'] = self._detect_contexts(content, pattern.context_rules)
        
        # Statistics
        result['stats'] = self._calculate_stats(result['elements'])
        
        return result
    
    def _parse_syntax_patterns(self, content: str, syntax_patterns: Dict) -> Dict[str, Any]:
        """Parse content using syntax patterns"""
        elements = {}
        
        for category, pattern_def in syntax_patterns.items():
            if isinstance(pattern_def, dict):
                if 'regex' in pattern_def:
                    matches = self._extract_with_pattern(content, pattern_def['regex'], 
                                                         pattern_def.get('capture_groups', []))
                    if matches:
                        elements[category] = matches
                
                elif 'patterns' in pattern_def:
                    for sub_pattern in pattern_def['patterns']:
                        if isinstance(sub_pattern, dict) and 'regex' in sub_pattern:
                            matches = self._extract_with_pattern(
                                content, sub_pattern['regex'],
                                sub_pattern.get('capture_groups', [])
                            )
                            if matches:
                                elements[f"{category}.{sub_pattern.get('type', 'sub')}"] = matches
            
            elif isinstance(pattern_def, list):
                all_matches = []
                for item in pattern_def:
                    if isinstance(item, dict) and 'regex' in item:
                        matches = self._extract_with_pattern(
                            content, item['regex'], 
                            item.get('capture_groups', [])
                        )
                        all_matches.extend(matches)
                if all_matches:
                    elements[category] = all_matches
        
        return elements
    
    def _extract_with_pattern(self, content: str, pattern: str, 
                             capture_groups: List[str]) -> List[Dict]:
        """Extract matches with named capture groups"""
        matches = []
        try:
            for match in re.finditer(pattern, content, re.DOTALL | re.MULTILINE):
                match_dict = {'match': match.group(0)}
                
                # Add numbered captures
                for i in range(1, len(match.groups()) + 1):
                    if match.group(i):
                        match_dict[f'group_{i}'] = match.group(i)
                
                # Add named captures if provided
                for i, group_name in enumerate(capture_groups, 1):
                    if i <= len(match.groups()) and match.group(i):
                        match_dict[group_name] = match.group(i)
                
                matches.append(match_dict)
        except re.error as e:
            print(f"Regex error in pattern {pattern}: {e}")
        
        return matches
    
    def _parse_flow_patterns(self, content: str, flow_patterns: Dict) -> List[Dict]:
        """Parse flow/structural patterns"""
        flows = []
        
        for flow_name, flow_def in flow_patterns.items():
            if isinstance(flow_def, dict) and 'pattern' in flow_def:
                pattern = flow_def['pattern']
                description = flow_def.get('description', flow_name)
                
                try:
                    for match in re.finditer(pattern, content, re.DOTALL):
                        flows.append({
                            'type': flow_name,
                            'description': description,
                            'content': match.group(0),
                            'position': match.start()
                        })
                except re.error:
                    pass
        
        # Sort by position
        flows.sort(key=lambda x: x['position'])
        return flows
    
    def _detect_contexts(self, content: str, context_rules: Dict) -> List[Dict]:
        """Detect different contexts in the code"""
        contexts = []
        lines = content.split('\n')
        
        for context_name, context_def in context_rules.items():
            if isinstance(context_def, dict):
                description = context_def.get('description', context_name)
                patterns = context_def.get('patterns', [])
                
                context_matches = []
                for i, line in enumerate(lines):
                    for pattern in patterns:
                        if re.search(pattern, line):
                            context_matches.append({
                                'line': i + 1,
                                'content': line.strip(),
                                'pattern': pattern
                            })
                            break
                
                if context_matches:
                    contexts.append({
                        'name': context_name,
                        'description': description,
                        'matches': context_matches
                    })
        
        return contexts
    
    def _parse_generic(self, content: str, language: str) -> Dict[str, Any]:
        """Generic parsing when no specific pattern is available"""
        return {
            'language': language,
            'raw': content,
            'elements': {'generic': [{'match': content[:100] + '...'}]},
            'stats': {'lines': len(content.split('\n')), 'chars': len(content)}
        }
    
    def _calculate_stats(self, elements: Dict) -> Dict[str, int]:
        """Calculate statistics about parsed elements"""
        stats = {}
        for category, matches in elements.items():
            if isinstance(matches, list):
                stats[f"{category}_count"] = len(matches)
            elif isinstance(matches, dict):
                stats[f"{category}_count"] = len(matches)
        
        return stats

class SmartExtractor:
    """Smart extractor that understands multi-language embedded code"""
    
    def __init__(self, pattern_registry: PatternRegistry):
        self.registry = pattern_registry
        self.parser = PatternDrivenParser(pattern_registry)
    
    def extract_from_file(self, filepath: str) -> Dict[str, Any]:
        """Extract all language elements from a file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            result = {
                'file': filepath,
                'extension': Path(filepath).suffix.lower(),
                'size': len(content),
                'primary_language': '',
                'embedded_languages': {},
                'extracted_elements': {},
                'flow_analysis': {}
            }
            
            # Detect primary language
            primary_lang, confidence = self.registry.detect_language(content, filepath)
            result['primary_language'] = primary_lang
            result['confidence'] = confidence
            
            # Parse primary content
            parsed = self.parser.parse_content(content, primary_lang)
            result['extracted_elements'][primary_lang] = parsed['elements']
            result['flow_analysis'][primary_lang] = parsed['flow']
            
            # Extract embedded languages
            embedded = self._extract_embedded_languages(content, primary_lang)
            result['embedded_languages'] = embedded
            
            # Cross-language references
            result['cross_references'] = self._find_cross_references(
                parsed['elements'], embedded
            )
            
            return result
            
        except Exception as e:
            return {
                'file': filepath,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _extract_embedded_languages(self, content: str, primary_lang: str) -> Dict[str, Any]:
        """Extract embedded languages within primary content"""
        embedded = {}
        
        # PHP in HTML
        if primary_lang == 'html':
            php_patterns = [
                (r'<\?php\s+(.*?)\?>', 'php'),
                (r'<\?=\s*(.*?)\?>', 'php_echo')
            ]
            for pattern, lang_type in php_patterns:
                for match in re.finditer(pattern, content, re.DOTALL):
                    if match.group(1):
                        embedded.setdefault('php', []).append({
                            'type': lang_type,
                            'content': match.group(1),
                            'position': match.start()
                        })
        
        # JavaScript in HTML
        if primary_lang == 'html':
            js_patterns = [
                (r'<script[^>]*>(.*?)</script>', 'javascript'),
                (r'on\w+="([^"]+)"', 'javascript_inline')
            ]
            for pattern, lang_type in js_patterns:
                for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                    if match.group(1):
                        embedded.setdefault('javascript', []).append({
                            'type': lang_type,
                            'content': match.group(1),
                            'position': match.start()
                        })
        
        # CSS in HTML
        if primary_lang == 'html':
            css_patterns = [
                (r'<style[^>]*>(.*?)</style>', 'css'),
                (r'style="([^"]+)"', 'css_inline')
            ]
            for pattern, lang_type in css_patterns:
                for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                    if match.group(1):
                        embedded.setdefault('css', []).append({
                            'type': lang_type,
                            'content': match.group(1),
                            'position': match.start()
                        })
        
        # SQL in PHP
        if primary_lang == 'php':
            sql_patterns = [
                (r'mysql(i)?_query\s*\(\s*["\']([^"\']+)["\']', 'mysql'),
                (r'PDO::(query|exec|prepare)\s*\(\s*["\']([^"\']+)["\']', 'pdo')
            ]
            for pattern, lang_type in sql_patterns:
                for match in re.finditer(pattern, content, re.DOTALL):
                    if match.lastindex and match.group(match.lastindex):
                        embedded.setdefault('sql', []).append({
                            'type': lang_type,
                            'content': match.group(match.lastindex),
                            'position': match.start()
                        })
        
        # Parse each embedded language
        for lang, snippets in embedded.items():
            for snippet in snippets:
                parsed = self.parser.parse_content(snippet['content'], lang)
                snippet['parsed'] = parsed
        
        return embedded
    
    def _find_cross_references(self, primary_elements: Dict, 
                               embedded: Dict) -> List[Dict]:
        """Find cross-references between languages"""
        references = []
        
        # Find variables that are shared
        if 'variables' in primary_elements:
            primary_vars = set()
            for var_match in primary_elements.get('variables', []):
                if 'name' in var_match:
                    primary_vars.add(var_match['name'])
            
            # Check if these variables appear in embedded code
            for lang, snippets in embedded.items():
                for snippet in snippets:
                    if 'parsed' in snippet and 'elements' in snippet['parsed']:
                        if 'variables' in snippet['parsed']['elements']:
                            for var_match in snippet['parsed']['elements']['variables']:
                                if 'name' in var_match and var_match['name'] in primary_vars:
                                    references.append({
                                        'type': 'variable_reference',
                                        'variable': var_match['name'],
                                        'from_language': 'primary',
                                        'to_language': lang,
                                        'snippet': snippet['content'][:100]
                                    })
        
        return references

class GodTierConverter:
    """Advanced converter with pattern-based transformations"""
    
    def __init__(self, pattern_registry: PatternRegistry):
        self.registry = pattern_registry
    
    def convert(self, content: str, from_lang: str, to_lang: str, 
                options: Dict = None) -> Dict[str, Any]:
        """Convert code from one language to another"""
        options = options or {}
        
        # Parse source
        parser = PatternDrivenParser(self.registry)
        parsed = parser.parse_content(content, from_lang)
        
        # Get conversion patterns
        from_pattern = self.registry.get_pattern(from_lang)
        to_pattern = self.registry.get_pattern(to_lang)
        
        if not from_pattern or not to_pattern:
            return {
                'success': False,
                'error': f'Missing patterns for {from_lang} or {to_lang}',
                'original': content
            }
        
        # Apply conversions
        converted = self._apply_conversion_rules(
            parsed, from_pattern, to_pattern, options
        )
        
        return {
            'success': True,
            'from': from_lang,
            'to': to_lang,
            'original': content,
            'converted': converted,
            'warnings': converted.get('warnings', []),
            'stats': converted.get('stats', {})
        }
    
    def _apply_conversion_rules(self, parsed: Dict, 
                                from_pattern: LanguagePattern,
                                to_pattern: LanguagePattern,
                                options: Dict) -> Dict:
        """Apply pattern-based conversion rules"""
        result = {
            'code': '',
            'warnings': [],
            'stats': {}
        }
        
        # Get conversion patterns
        conversion_key = f"to_{to_pattern.language}"
        conversion_rules = from_pattern.conversion_patterns.get(conversion_key, {})
        
        if not conversion_rules:
            # Try reverse lookup
            for key, rules in to_pattern.conversion_patterns.items():
                if key.startswith('to_'):
                    from_key = key[3:]  # Remove 'to_'
                    if from_key == from_pattern.language:
                        # We need to invert these rules
                        result['warnings'].append(
                            f"No direct conversion rules found, attempting reverse mapping"
                        )
                        conversion_rules = self._invert_rules(rules)
                        break
        
        if not conversion_rules:
            result['code'] = self._generic_conversion(parsed, from_pattern, to_pattern)
            result['warnings'].append(
                f"No conversion patterns found, using generic conversion"
            )
        else:
            result['code'] = self._pattern_based_conversion(
                parsed['raw'], conversion_rules, options
            )
        
        # Calculate statistics
        result['stats'] = {
            'original_length': len(parsed['raw']),
            'converted_length': len(result['code']),
            'elements_converted': len(parsed.get('elements', {})),
            'line_count': len(result['code'].split('\n'))
        }
        
        return result
    
    def _pattern_based_conversion(self, content: str, 
                                 conversion_rules: Dict, 
                                 options: Dict) -> str:
        """Convert using pattern-based rules"""
        converted = content
        
        for rule_name, rule_pattern in conversion_rules.items():
            if isinstance(rule_pattern, str):
                try:
                    # Simple pattern replacement
                    if re.search(rule_pattern, converted):
                        # This is a simplification - real implementation would be more complex
                        converted = re.sub(rule_pattern, 
                                          self._get_replacement(rule_name, rule_pattern),
                                          converted)
                except re.error:
                    pass
        
        return converted
    
    def _generic_conversion(self, parsed: Dict, 
                           from_pattern: LanguagePattern,
                           to_pattern: LanguagePattern) -> str:
        """Generic conversion when no patterns are available"""
        elements = parsed.get('elements', {})
        
        if to_pattern.language == 'html':
            return self._to_html_generic(elements, parsed['raw'])
        elif to_pattern.language == 'php':
            return self._to_php_generic(elements, parsed['raw'])
        elif to_pattern.language == 'javascript':
            return self._to_js_generic(elements, parsed['raw'])
        else:
            return f"// Converted from {from_pattern.language} to {to_pattern.language}\n" + parsed['raw']
    
    def _to_html_generic(self, elements: Dict, original: str) -> str:
        """Generic conversion to HTML"""
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<meta charset="UTF-8">',
            f'<title>Converted from code</title>',
            '</head>',
            '<body>',
            '<div class="converted-content">',
            f'<pre><code>{html.escape(original)}</code></pre>'
        ]
        
        # Add extracted elements as HTML
        for category, matches in elements.items():
            if matches:
                html_parts.append(f'<h3>{category}</h3>')
                html_parts.append('<ul>')
                for match in matches[:10]:  # Limit to 10
                    if isinstance(match, dict) and 'match' in match:
                        html_parts.append(f'<li>{html.escape(str(match["match"][:100]))}</li>')
                html_parts.append('</ul>')
        
        html_parts.extend(['</div>', '</body>', '</html>'])
        return '\n'.join(html_parts)
    
    def _to_php_generic(self, elements: Dict, original: str) -> str:
        """Generic conversion to PHP"""
        php_parts = ['<?php']
        php_parts.append('// Generated from code conversion')
        
        if 'variables' in elements:
            php_parts.append('// Variables found:')
            for var_match in elements['variables']:
                if 'name' in var_match:
                    php_parts.append(f'${var_match["name"]} = null; // extracted')
        
        if 'functions' in elements:
            php_parts.append('// Function stubs:')
            for func_match in elements['functions']:
                if 'name' in func_match:
                    php_parts.append(f'function {func_match["name"]}() {{')
                    php_parts.append('    // TODO: Implement')
                    php_parts.append('}')
        
        php_parts.append('?>')
        return '\n'.join(php_parts)
    
    def _to_js_generic(self, elements: Dict, original: str) -> str:
        """Generic conversion to JavaScript"""
        js_parts = ['// Generated from code conversion']
        
        if 'variables' in elements:
            js_parts.append('// Variables:')
            for var_match in elements['variables']:
                if 'name' in var_match:
                    js_parts.append(f'let {var_match["name"]} = null;')
        
        if 'functions' in elements:
            js_parts.append('// Functions:')
            for func_match in elements['functions']:
                if 'name' in func_match:
                    js_parts.append(f'function {func_match["name"]}() {{')
                    js_parts.append('    // TODO: Implement')
                    js_parts.append('}')
        
        return '\n'.join(js_parts)
    
    def _get_replacement(self, rule_name: str, pattern: str) -> str:
        """Get replacement string for a pattern"""
        replacements = {
            'variable': '\\$\\1',
            'function': 'function \\1',
            'echo_statement': 'echo \\1;',
            'php_tags': '<?php \\1 ?>'
        }
        return replacements.get(rule_name, '\\0')
    
    def _invert_rules(self, rules: Dict) -> Dict:
        """Invert conversion rules (to_X -> from_X)"""
        inverted = {}
        # This would require complex pattern inversion
        # For now, return empty dict
        return inverted

class MixAssCore:
    """Core orchestrator for mix_ass functionality"""
    
    def __init__(self, pattern_dir: str = "patterns"):
        self.registry = PatternRegistry(pattern_dir)
        self.extractor = SmartExtractor(self.registry)
        self.converter = GodTierConverter(self.registry)
        self.parser = PatternDrivenParser(self.registry)
    
    def analyze_file(self, filepath: str, output_format: str = 'json') -> Dict:
        """Analyze a file with full extraction"""
        result = self.extractor.extract_from_file(filepath)
        
        if output_format == 'yaml':
            import yaml
            return yaml.dump(result, default_flow_style=False)
        elif output_format == 'text':
            return self._format_text_result(result)
        else:  # json
            return json.dumps(result, indent=2, default=str)
    
    def convert_file(self, filepath: str, from_lang: str, to_lang: str, 
                     output_file: str = None) -> Dict:
        """Convert a file from one language to another"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = self.converter.convert(content, from_lang, to_lang)
        
        if output_file and result['success']:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['converted']['code'])
        
        return result
    
    def extract_patterns(self, filepath: str, language: str = None) -> Dict:
        """Extract and learn new patterns from a file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not language:
            language, _ = self.registry.detect_language(content, filepath)
        
        # Parse to discover patterns
        parsed = self.parser.parse_content(content, language)
        
        # Generate pattern suggestions
        suggestions = self._generate_pattern_suggestions(parsed, language)
        
        return {
            'file': filepath,
            'language': language,
            'suggested_patterns': suggestions,
            'statistics': parsed.get('stats', {})
        }
    
    def _generate_pattern_suggestions(self, parsed: Dict, language: str) -> List[Dict]:
        """Generate new pattern suggestions from parsed content"""
        suggestions = []
        elements = parsed.get('elements', {})
        
        for category, matches in elements.items():
            if matches and len(matches) > 2:  # Need multiple examples to suggest pattern
                # Analyze common patterns in matches
                sample_matches = [m.get('match', '') for m in matches[:5]]
                
                # Try to find common structure
                if all('=' in m for m in sample_matches):
                    suggestions.append({
                        'category': category,
                        'type': 'assignment_pattern',
                        'suggestion': f"Detected assignment pattern for {category}",
                        'sample': sample_matches[0]
                    })
                elif all('(' in m and ')' in m for m in sample_matches):
                    suggestions.append({
                        'category': category,
                        'type': 'function_pattern',
                        'suggestion': f"Detected function/method pattern for {category}",
                        'sample': sample_matches[0]
                    })
        
        return suggestions
    
    def _format_text_result(self, result: Dict) -> str:
        """Format result as human-readable text"""
        lines = []
        lines.append(f"File: {result.get('file', 'Unknown')}")
        lines.append(f"Primary Language: {result.get('primary_language', 'Unknown')}")
        lines.append(f"Confidence: {result.get('confidence', 0):.2f}")
        lines.append("")
        
        # Elements found
        elements = result.get('extracted_elements', {})
        for lang, lang_elements in elements.items():
            lines.append(f"{lang.upper()} Elements:")
            for category, matches in lang_elements.items():
                if isinstance(matches, list):
                    lines.append(f"  {category}: {len(matches)} found")
                elif isinstance(matches, dict):
                    lines.append(f"  {category}: {len(matches)} items")
            lines.append("")
        
        # Embedded languages
        embedded = result.get('embedded_languages', {})
        if embedded:
            lines.append("Embedded Languages:")
            for lang, snippets in embedded.items():
                lines.append(f"  {lang}: {len(snippets)} snippets")
        
        return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="MixAss - God-Tier Multi-Language Pattern-Driven Extractor & Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze agi_world_model_builder.ass --format json
  %(prog)s convert script.php --from php --to html --output script.html
  %(prog)s extract-patterns complex.js --language javascript
  %(prog)s batch-analyze ./project/ --output report.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a file')
    analyze_parser.add_argument('file', help='File to analyze')
    analyze_parser.add_argument('--format', choices=['json', 'yaml', 'text'], 
                               default='json', help='Output format')
    analyze_parser.add_argument('--pattern-dir', default='patterns',
                               help='Pattern directory')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between languages')
    convert_parser.add_argument('file', help='Input file')
    convert_parser.add_argument('--from', dest='from_lang', required=True,
                               help='Source language')
    convert_parser.add_argument('--to', dest='to_lang', required=True,
                               help='Target language')
    convert_parser.add_argument('--output', '-o', help='Output file')
    convert_parser.add_argument('--pattern-dir', default='patterns',
                               help='Pattern directory')
    
    # Extract patterns command
    pattern_parser = subparsers.add_parser('extract-patterns', 
                                          help='Extract patterns from file')
    pattern_parser.add_argument('file', help='File to analyze')
    pattern_parser.add_argument('--language', help='Language (auto-detect if not specified)')
    pattern_parser.add_argument('--pattern-dir', default='patterns',
                               help='Pattern directory')
    
    # Batch analyze command
    batch_parser = subparsers.add_parser('batch-analyze', 
                                        help='Analyze multiple files')
    batch_parser.add_argument('path', help='Directory or file pattern')
    batch_parser.add_argument('--output', '-o', required=True, help='Output file')
    batch_parser.add_argument('--pattern-dir', default='patterns',
                             help='Pattern directory')
    
    # List patterns command
    list_parser = subparsers.add_parser('list-patterns', 
                                       help='List available patterns')
    list_parser.add_argument('--pattern-dir', default='patterns',
                            help='Pattern directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize core
    core = MixAssCore(args.pattern_dir)
    
    if args.command == 'analyze':
        result = core.analyze_file(args.file, args.format)
        print(result)
    
    elif args.command == 'convert':
        result = core.convert_file(args.file, args.from_lang, args.to_lang, args.output)
        if result['success']:
            print(f"✅ Conversion successful!")
            if args.output:
                print(f"   Output saved to: {args.output}")
            else:
                print("\n" + result['converted']['code'])
        else:
            print(f"❌ Conversion failed: {result.get('error')}")
    
    elif args.command == 'extract-patterns':
        result = core.extract_patterns(args.file, args.language)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'batch-analyze':
        # Process multiple files
        results = []
        path = Path(args.path)
        
        if path.is_file():
            files = [path]
        else:
            # Get all supported files
            extensions = {ext for pattern in core.registry.patterns.values() 
                         for ext in pattern.extensions}
            files = []
            for ext in extensions:
                files.extend(path.glob(f"**/*{ext}"))
        
        print(f"Analyzing {len(files)} files...")
        for i, filepath in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] {filepath}")
            result = core.extractor.extract_from_file(str(filepath))
            results.append(result)
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Analysis complete. Results saved to: {args.output}")
    
    elif args.command == 'list-patterns':
        print("Available Language Patterns:")
        print("=" * 50)
        for lang_name, pattern in core.registry.patterns.items():
            print(f"Language: {lang_name}")
            print(f"  Extensions: {', '.join(pattern.extensions)}")
            print(f"  Syntax Categories: {len(pattern.syntax_patterns)}")
            print(f"  Flow Patterns: {len(pattern.flow_patterns)}")
            print()

if __name__ == '__main__':
    main()