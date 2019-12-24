#include <vector>
#include <Python.h>

#include "AdaGramModel.h"
#include "PyDoc.h"

using namespace std;

PyObject* gModule;

struct AGMObject
{
	PyObject_HEAD;
	ag::AdaGramModel<ag::Mode::hierarchical_softmax>* inst;

	static void dealloc(AGMObject* self)
	{
		if (self->inst)
		{
			delete self->inst;
		}
		Py_TYPE(self)->tp_free((PyObject*)self);
	}

	static int init(AGMObject *self, PyObject *args, PyObject *kwargs)
	{
		size_t emb_size = 300, max_prototypes = 5;
		float subsampling = 1e-4;
		size_t ns = 5;
        float alpha = 1e-1f, d = 0;
		size_t seed = std::random_device{}();
		static const char* kwlist[] = { "emb_size", "max_prototypes", "subsampling", "alpha", "d", "seed", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnfffn", (char**)kwlist,
			&emb_size, &max_prototypes, &subsampling, &alpha, &d, &seed)) return -1;
		try
		{
			auto* inst = new ag::AdaGramModel<ag::Mode::hierarchical_softmax>(emb_size, max_prototypes, alpha, d, subsampling, seed);
			self->inst = inst;
		}
		catch (const exception& e)
		{
			PyErr_SetString(PyExc_Exception, e.what());
			return -1;
		}
		return 0;
	}

	static PyObject* getVocabs(AGMObject *self, void* closure);
};

struct VocabDictObject
{
	PyObject_HEAD;
	AGMObject* parentObj;

	static void dealloc(VocabDictObject* self)
	{
		Py_XDECREF(self->parentObj);
		Py_TYPE(self)->tp_free((PyObject*)self);
	}

	static int init(VocabDictObject *self, PyObject *args, PyObject *kwargs)
	{
		static const char* kwlist[] = { "parent", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &self->parentObj)) return -1;
		Py_INCREF(self->parentObj);
		return 0;
	}

	static Py_ssize_t length(VocabDictObject* self)
	{
		try
		{
			return self->parentObj->inst->getVocabs().size();
		}
		catch (const exception& e)
		{
			PyErr_SetString(PyExc_Exception, e.what());
			return -1;
		}
	}

	static PyObject* getItem(VocabDictObject* self, Py_ssize_t key)
	{
		try
		{
			if (key < self->parentObj->inst->getVocabs().size()) return py::buildPyValue(self->parentObj->inst->getVocabs()[key]);
			PyErr_SetString(PyExc_IndexError, "");
			return nullptr;
		}
		catch (const bad_exception&)
		{
			return nullptr;
		}
		catch (const exception& e)
		{
			PyErr_SetString(PyExc_Exception, e.what());
			return nullptr;
		}
	}
};

static PyMethodDef AGM_methods[] =
{
	{ "load", (PyCFunction)CGM_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, CGM_load__doc__ },
	{ "save", (PyCFunction)CGM_save, METH_VARARGS | METH_KEYWORDS, CGM_save__doc__ },
	{ "build_vocab", (PyCFunction)CGM_buildVocab, METH_VARARGS | METH_KEYWORDS, CGM_build_vocab__doc__ },
	{ "train", (PyCFunction)CGM_train, METH_VARARGS | METH_KEYWORDS, CGM_train__doc__ },
	{ "most_similar", (PyCFunction)CGM_mostSimilar, METH_VARARGS | METH_KEYWORDS, CGM_most_similar__doc__ },
	{ "similarity", (PyCFunction)CGM_similarity, METH_VARARGS | METH_KEYWORDS, CGM_similarity__doc__ },
	{ "embedding", (PyCFunction)CGM_getEmbedding, METH_VARARGS | METH_KEYWORDS, CGM_embedding__doc__ },
	{ nullptr }
};

static PyGetSetDef AGM_getseters[] = {
	{ (char*)"d", (getter)CGM_getD, nullptr, (char*)"embedding dimension", NULL },
	{ (char*)"r", (getter)CGM_getR, nullptr, (char*)"chebyshev approximation order", NULL },
	{ (char*)"zeta", (getter)CGM_getZeta, nullptr, (char*)"zeta, mixing factor", NULL },
	{ (char*)"lambda_v", (getter)CGM_getLambda, nullptr, (char*)"lambda", NULL },
	{ (char*)"min_time", (getter)CGM_getMinPoint, nullptr, (char*)"time range", NULL },
	{ (char*)"max_time", (getter)CGM_getMaxPoint, nullptr, (char*)"time range", NULL },
	{ (char*)"vocabs", (getter)CGMObject::getVocabs, nullptr, (char*)"vocabularies in the model", NULL },
	{ (char*)"padding", (getter)CGM_getPadding, (setter)CGM_setPadding, (char*)"padding", NULL },
	{ (char*)"tp_bias", (getter)CGM_getTPBias, (setter)CGM_setTPBias, (char*)"bias of whole temporal distribution", NULL },
	{ (char*)"tp_threshold", (getter)CGM_getTPThreshold, (setter)CGM_setTPThreshold, (char*)"filtering threshold on temporal probability", NULL },
	{ nullptr },
};

static PyTypeObject AGM_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"adagram.Adagram",             /* tp_name */
	sizeof(CGMObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)AGMObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	AGM___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	AGM_methods,             /* tp_methods */
	0,						 /* tp_members */
	AGM_getseters,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)AGMObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

static PySequenceMethods VocabDict_seqs = {
	(lenfunc)VocabDictObject::length,
	0,
	0,
	(ssizeargfunc)VocabDictObject::getItem,
};

static PyTypeObject VocabDict_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"adagram._Vocabs",             /* tp_name */
	sizeof(VocabDictObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)VocabDictObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	&VocabDict_seqs,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,  /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	VocabDict___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	0,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)VocabDictObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};


PyMODINIT_FUNC MODULE_NAME()
{
	static PyModuleDef mod =
	{
		PyModuleDef_HEAD_INIT,
		"adagram",
		"AdaGram(Adaptive Skip-gram) Module for Python",
		-1,
		nullptr,
	};

	gModule = PyModule_Create(&mod);
	if (!gModule) return nullptr;

    if(PyType_Ready(&AGM_type) < 0) return nullptr;
    Py_INCREF(&AGM_type);
    PyModule_AddObject(gModule, "Adagram", (PyObject*)&AGM_type);

    if(PyType_Ready(&VocabDict_type) < 0) return nullptr;
    Py_INCREF(&VocabDict_type);
    PyModule_AddObject(gModule, "_Vocabs", (PyObject*)&VocabDict_type);

#ifdef __AVX2__
	PyModule_AddStringConstant(gModule, "isa", "avx2");
#elif defined(__AVX__)
	PyModule_AddStringConstant(gModule, "isa", "avx");
#elif defined(__SSE2__) || defined(__x86_64__) || defined(_WIN64)
	PyModule_AddStringConstant(gModule, "isa", "sse2");
#else
	PyModule_AddStringConstant(gModule, "isa", "none");
#endif
	return gModule;
}
