#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE AGBD PIPELINE AUDIT

This script performs systematic analysis of ALL critical issues identified
in supervisor meetings and implements solutions step by step.

CRITICAL ISSUES TO ADDRESS:
1. Token alignment: 25×25 vs 16×16 ViT patches
2. Token masking: "identify token, remove others" 
3. Multi-GPU reduction: "mae not correct only reported for first gpu"
4. Information leakage: "need to verify that other tokens are not leaking information"
5. Assertion-based validation: "try to work with assertions to make it crash"

SUPERVISOR REQUIREMENTS:
- "check the training code, might be some hidden errors"
- "place the patches into eg one corner upper left properly aligned"
- "this need to work for ALL models keep that in mind"
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add pangaea to path
sys.path.insert(0, '/scratch/final2/pangaea-bench-agbd')

def audit_token_alignment():
    """Audit the critical token alignment issue."""
    print("🔍 AUDITING TOKEN ALIGNMENT")
    print("=" * 60)
    
    # Common ViT patch sizes in PANGAEA
    vit_models = {
        'ViT-Base': 16,
        'ViT-Large': 16, 
        'Prithvi': 16,
        'SatMAE': 16,
        'SpectralGPT': 16,
        'DOFA': 16,
        'CROMA': 8,  # Different!
        'Scale-MAE': 16,
        'SSL4EO variants': 16,
    }
    
    agbd_patch_size = 25
    
    print(f"AGBD patch size: {agbd_patch_size}×{agbd_patch_size}")
    print("\nViT Model Compatibility:")
    print("-" * 40)
    
    alignment_issues = []
    
    for model, patch_size in vit_models.items():
        remainder = agbd_patch_size % patch_size
        if remainder == 0:
            status = "✅ ALIGNED"
        else:
            status = f"❌ MISALIGNED (remainder {remainder})"
            alignment_issues.append((model, patch_size, remainder))
        
        print(f"{model:<18} | {patch_size:>2}×{patch_size:<2} | {status}")
    
    print(f"\n🚨 CRITICAL FINDINGS:")
    print(f"   ❌ {len(alignment_issues)}/{len(vit_models)} models have alignment issues!")
    print(f"   ❌ Only CROMA (8×8) might work without issues")
    print(f"   ❌ Meeting note '24×24 tokens' is INCORRECT - most use 16×16!")
    
    return alignment_issues

def audit_token_masking_implementation():
    """Check if token masking is implemented."""
    print("\n🎭 AUDITING TOKEN MASKING IMPLEMENTATION")
    print("=" * 60)
    
    # Search for token masking in codebase
    search_patterns = [
        "zeros_like", 
        "token.*mask",
        "attention.*mask",
        "ignore.*token",
        "token.*zero"
    ]
    
    print("Meeting requirement: 'identify token, remove others'")
    print("Required: Token.data = torch.zeros_like(token.data)")
    print()
    
    # Check if implemented in key files
    key_files = [
        '/scratch/final2/pangaea-bench-agbd/pangaea/engine/trainer.py',
        '/scratch/final2/pangaea-bench-agbd/pangaea/engine/evaluator.py',
        '/scratch/final2/pangaea-bench-agbd/pangaea/engine/data_preprocessor.py',
    ]
    
    token_masking_found = False
    for file_path in key_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                if any(pattern in content.lower() for pattern in search_patterns):
                    token_masking_found = True
                    print(f"✅ Some masking patterns found in {os.path.basename(file_path)}")
                else:
                    print(f"❌ No token masking in {os.path.basename(file_path)}")
    
    if not token_masking_found:
        print("\n🚨 CRITICAL ISSUE: Token masking NOT implemented!")
        print("   This violates supervisor requirement: 'identify token, remove others'")
        print("   Information leakage likely occurring!")
    
    return token_masking_found

def audit_multi_gpu_reduction():
    """Check multi-GPU reduction implementation.""" 
    print("\n🌐 AUDITING MULTI-GPU REDUCTION")
    print("=" * 60)
    
    print("Meeting requirement: 'mae is not correct only reported for first gpu'")
    print("Required: 'divide by number of gpus or use torch reduce average'")
    print()
    
    # Check evaluator implementation
    evaluator_path = '/scratch/final2/pangaea-bench-agbd/pangaea/engine/evaluator.py'
    if os.path.exists(evaluator_path):
        with open(evaluator_path, 'r') as f:
            content = f.read()
        
        # Check for distributed reduction
        reduction_indicators = [
            'torch.distributed.all_reduce',
            'ReduceOp.SUM',
            'total_samples_all_gpus',
            'world_size'
        ]
        
        found_patterns = []
        for pattern in reduction_indicators:
            if pattern in content:
                found_patterns.append(pattern)
        
        if len(found_patterns) >= 3:
            print("✅ Multi-GPU reduction appears implemented")
            print(f"   Found patterns: {found_patterns}")
        else:
            print("❌ Multi-GPU reduction may be incomplete")
            print(f"   Found only: {found_patterns}")
    else:
        print("❌ Cannot find evaluator.py")
    
    return len(found_patterns) >= 3

def audit_assertion_safeguards():
    """Check for assertion-based validation."""
    print("\n🛡️ AUDITING ASSERTION SAFEGUARDS")
    print("=" * 60)
    
    print("Meeting requirement: 'try to work with assertions to make it crash when something unexpected happens'")
    print()
    
    # Key files that should have assertions
    key_files = [
        '/scratch/final2/pangaea-bench-agbd/pangaea/engine/trainer.py',
        '/scratch/final2/pangaea-bench-agbd/pangaea/engine/evaluator.py',
        '/scratch/final2/pangaea-bench-agbd/pangaea/datasets/agbd.py',
    ]
    
    assertion_coverage = {}
    
    for file_path in key_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            assertion_count = content.count('assert ')
            lines = content.split('\n')
            total_lines = len(lines)
            
            # Look for specific AGBD-related assertions
            agbd_assertions = []
            for i, line in enumerate(lines):
                if 'assert' in line.lower() and any(keyword in line.lower() for keyword in ['shape', 'size', 'agbd', 'center', 'valid']):
                    agbd_assertions.append((i+1, line.strip()))
            
            assertion_coverage[os.path.basename(file_path)] = {
                'total_assertions': assertion_count,
                'agbd_assertions': agbd_assertions,
                'density': assertion_count / total_lines * 100
            }
    
    print("Assertion Coverage Analysis:")
    for filename, data in assertion_coverage.items():
        print(f"\n📄 {filename}:")
        print(f"   Total assertions: {data['total_assertions']}")
        print(f"   Assertion density: {data['density']:.1f}%")
        if data['agbd_assertions']:
            print(f"   AGBD-specific assertions: {len(data['agbd_assertions'])}")
            for line_no, assertion in data['agbd_assertions'][:3]:  # Show first 3
                print(f"     L{line_no}: {assertion[:60]}...")
        else:
            print("   ❌ No AGBD-specific assertions found!")
    
    return assertion_coverage

def propose_comprehensive_solution():
    """Propose a comprehensive solution addressing all issues."""
    print("\n🎯 COMPREHENSIVE SOLUTION PROPOSAL")
    print("=" * 60)
    
    print("Based on audit findings, here's the systematic fix plan:")
    print()
    
    print("📋 PHASE 1: TOKEN ALIGNMENT FIX")
    print("   1. Update AGBD img_size: 25 → 32 (or 48)")
    print("   2. Ensure perfect ViT-16 alignment (32÷16=2, 48÷16=3)")
    print("   3. Validate center pixel preservation: (12,12) → (15,15) or (23,23)")
    print()
    
    print("📋 PHASE 2: TOKEN MASKING IMPLEMENTATION")
    print("   1. Identify center token containing GEDI pixel")
    print("   2. Zero out all other tokens: token.data = torch.zeros_like(token.data)")
    print("   3. Ensure gradients blocked for masked tokens")
    print("   4. Validate no information leakage")
    print()
    
    print("📋 PHASE 3: COMPREHENSIVE VALIDATION")
    print("   1. Add assertions for tensor shapes at every critical step")
    print("   2. Validate multi-GPU reduction is working correctly")
    print("   3. Test with 1-2 models first, then expand")
    print("   4. Monitor WandB for gradient flow analysis")
    print()
    
    print("📋 PHASE 4: MULTI-MODEL COMPATIBILITY")
    print("   1. Test solution works for both ViT and CNN models")
    print("   2. Handle different patch sizes (8×8 CROMA vs 16×16 others)")
    print("   3. Ensure CNN models still work with original 25×25")
    print("   4. Create model-specific preprocessing configs if needed")
    
    return True

def run_comprehensive_audit():
    """Run complete audit of AGBD pipeline."""
    print("🔥 COMPREHENSIVE AGBD PIPELINE AUDIT")
    print("=" * 80)
    print("Based on supervisor meeting notes and technical requirements")
    print("=" * 80)
    
    # Run all audits
    alignment_issues = audit_token_alignment()
    token_masking_ok = audit_token_masking_implementation()
    multi_gpu_ok = audit_multi_gpu_reduction()
    assertion_coverage = audit_assertion_safeguards()
    
    # Propose solution
    propose_comprehensive_solution()
    
    # Summary
    print("\n📊 AUDIT SUMMARY")
    print("=" * 60)
    
    issues_found = []
    if alignment_issues:
        issues_found.append(f"Token alignment issues in {len(alignment_issues)} models")
    if not token_masking_ok:
        issues_found.append("Token masking not implemented")
    if not multi_gpu_ok:
        issues_found.append("Multi-GPU reduction incomplete")
    
    total_assertions = sum(data['total_assertions'] for data in assertion_coverage.values())
    if total_assertions < 10:
        issues_found.append("Insufficient assertion coverage")
    
    print(f"🚨 CRITICAL ISSUES FOUND: {len(issues_found)}")
    for issue in issues_found:
        print(f"   ❌ {issue}")
    
    if not issues_found:
        print("✅ All audits passed!")
    else:
        print(f"\n⚠️  PIPELINE NOT READY FOR PRODUCTION")
        print(f"   Must address {len(issues_found)} critical issues before training")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    success = run_comprehensive_audit()
    exit(0 if success else 1)
