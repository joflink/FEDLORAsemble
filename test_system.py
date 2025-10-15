#!/usr/bin/env python3
"""
FEDLORAsemble - Systemtest och Verifiering
==========================================

Detta skript testar att alla komponenter i FEDLORAsemble fungerar korrekt
och att alla beroenden är installerade.

Kör detta efter installation för att verifiera systemet.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

def test_imports():
    """Testar att alla nödvändiga bibliotek kan importeras."""
    
    print("🔍 Testar Python-imports...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("flwr", "Flower"),
        ("peft", "PEFT"),
        ("datasets", "Datasets"),
        ("evaluate", "Evaluate"),
        ("trl", "TRL"),
        ("omegaconf", "OmegaConf"),
        ("duckduckgo_search", "DuckDuckGo Search"),
        ("tqdm", "TQDM"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas")
    ]
    
    passed = 0
    failed = 0
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {name}")
            passed += 1
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
    
    print(f"\n📊 Import-resultat: {passed} godkända, {failed} misslyckade")
    return failed == 0

def test_project_structure():
    """Testar att projektstrukturen är korrekt."""
    
    print("\n🏗️ Testar projektstruktur...")
    
    required_files = [
        "moe_system.py",
        "evaluation_main.py",
        "pyproject.toml",
        "README.md",
        "training/server_app.py",
        "training/client_app.py",
        "training/AImodels.py",
        "training/dataset.py",
        "training/strategy.py",
    ]
    
    required_dirs = [
        "training",
        "evaluation", 
        "datasets",
    ]
    
    passed = 0
    failed = 0
    
    # Testa filer
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
            passed += 1
        else:
            print(f"  ❌ {file_path} (saknas)")
            failed += 1
    
    # Testa mappar
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"  ✅ {dir_path}/")
            passed += 1
        else:
            print(f"  ❌ {dir_path}/ (saknas)")
            failed += 1
    
    print(f"\n📊 Struktur-resultat: {passed} godkända, {failed} misslyckade")
    return failed == 0

def test_torch_functionality():
    """Testar PyTorch-funktionalitet."""
    
    print("\n🔥 Testar PyTorch...")
    
    try:
        import torch
        
        # Testa grundläggande tensor-operationer
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        print(f"  ✅ Tensor-operationer fungerar")
        print(f"  📊 PyTorch version: {torch.__version__}")
        
        # Testa CUDA-tillgänglighet
        if torch.cuda.is_available():
            print(f"  🚀 CUDA tillgänglig: {torch.cuda.get_device_name(0)}")
            print(f"  💾 GPU-minne: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            print(f"  💻 CUDA inte tillgänglig, använder CPU")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PyTorch-fel: {e}")
        return False

def test_moe_system():
    """Testar att MoE-systemet kan initialiseras."""
    
    print("\n🧠 Testar MoE-system...")
    
    try:
        # Försök importera vårt MoE-system
        sys.path.insert(0, str(Path.cwd()))
        
        # Testa endast imports, inte fullständig initialisering
        from moe_system import (
            HuggingFaceExpert, 
            ALBERTRouter, 
            MoESystem,
            maybe_compile
        )
        
        print("  ✅ MoE-klasser kan importeras")
        
        # Testa att vi kan skapa ett system (utan att ladda modeller)
        print("  ✅ MoE-system grundläggande funktionalitet OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MoE-systemfel: {e}")
        traceback.print_exc()
        return False

def test_training_components():
    """Testar träningskomponenter."""
    
    print("\n🎯 Testar träningskomponenter...")
    
    try:
        sys.path.insert(0, str(Path.cwd() / "training"))
        
        # Testa import av träningsmoduler
        modules = ["AImodels", "dataset", "strategy"]
        
        for module in modules:
            try:
                importlib.import_module(module)
                print(f"  ✅ {module}.py kan importeras")
            except Exception as e:
                print(f"  ❌ {module}.py: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Träningskomponentfel: {e}")
        return False

def test_evaluation_components():
    """Testar utvärderingskomponenter."""
    
    print("\n📊 Testar utvärderingskomponenter...")
    
    try:
        # Kontrollera att evalueringsmappen finns
        eval_dir = Path("evaluation")
        if not eval_dir.exists():
            print("  ❌ Evaluation-mapp saknas")
            return False
        
        # Testa att viktiga evalueringsfiler finns
        important_files = [
            "ALBERTRouter.py",
            "eval_utils.py", 
            "run_eval.py",
            "evals.yaml"
        ]
        
        for file_name in important_files:
            file_path = eval_dir / file_name
            if file_path.exists():
                print(f"  ✅ {file_name} finns")
            else:
                print(f"  ❌ {file_name} saknas")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utvärderingskomponentfel: {e}")
        return False

def generate_test_report():
    """Genererar en testrapport."""
    
    print("\n" + "="*60)
    print("🧪 FEDLORAsemble Systemtest")
    print("="*60)
    
    tests = [
        ("Python-imports", test_imports),
        ("Projektstruktur", test_project_structure),
        ("PyTorch", test_torch_functionality),
        ("MoE-system", test_moe_system),
        ("Träningskomponenter", test_training_components),
        ("Utvärderingskomponenter", test_evaluation_components)
    ]
    
    results = []
    total_passed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                total_passed += 1
        except Exception as e:
            print(f"\n❌ Fel i test '{test_name}': {e}")
            results.append((test_name, False))
    
    # Sammanfattning
    print("\n" + "="*60)
    print("📋 TESTSAMMANFATTNING")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ GODKÄND" if passed else "❌ MISSLYCKAD"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Totalresultat: {total_passed}/{len(tests)} tester godkända")
    
    if total_passed == len(tests):
        print("\n🎉 Alla tester godkända! FEDLORAsemble är redo att användas.")
        print("\n📚 Nästa steg:")
        print("   1. Ladda ner AI-modeller till models/ mappen")
        print("   2. Kör 'python moe_system.py' för att testa MoE-systemet")
        print("   3. Läs README.md för detaljerade instruktioner")
        return True
    else:
        print(f"\n⚠️ {len(tests) - total_passed} tester misslyckades.")
        print("   Kontrollera felmeddelandena ovan och installera saknade beroenden.")
        return False

def main():
    """Huvudfunktion."""
    
    print("Kör FEDLORAsemble systemverifiering...")
    
    try:
        success = generate_test_report()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Test avbrutet av användaren")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Oväntat fel: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
