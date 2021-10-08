Tab-separated format used by the DGEM model
   premise    hypothesis      label   hypothesis Open IE structure
where the "hypothesis Open IE structure" uses the format:
  $$$-separated tuples with <>-separated fields within each tuple
Each tuple should have 2+ fields (at least a subject + predicate)

Example Hypothesis: Most fossils are found in sedimentary rocks because organisms can be preserved in sedimentary rock.

Open IE Tuple:
(Most fossils; are found; L:in sedimentary rocks; because organisms can be preserved in sedimentary rock)
(organisms; can be preserved; L:in sedimentary rock)

Hypothesis Open IE structure:
Most fossils<>are found<>in sedimentary rocks<>because organisms can be preserved in sedimentary rock$$$organisms<>can be preserved<>in sedimentary rock


