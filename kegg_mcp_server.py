#!/usr/bin/env python3
"""
KEGG MCP Server (Python)

Model Context Protocol server for KEGG (Kyoto Encyclopedia of Genes and Genomes) database access.
Uses Streamable HTTP transport for web-based access.
"""

import asyncio
import json
import os
import re
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("KEGG MCP Server")

# KEGG API base URL
KEGG_API_BASE = "https://rest.kegg.jp"

# HTTP client for KEGG API
http_client = httpx.AsyncClient(
    base_url=KEGG_API_BASE,
    timeout=30.0,
    headers={
        "User-Agent": "KEGG-MCP-Server/1.0.0",
        "Accept": "text/plain",
    },
)


def parse_kegg_entry(data: str) -> dict[str, Any]:
    """Parse KEGG entry data into structured format."""
    lines = data.split("\n")
    result: dict[str, Any] = {}

    for line in lines:
        if line.startswith("ENTRY"):
            parts = line.split()
            if len(parts) >= 3:
                result["entry"] = parts[1]
                result["type"] = parts[2]
        elif line.startswith("NAME"):
            result["name"] = line[12:].strip()
        elif line.startswith("DEFINITION"):
            result["definition"] = line[12:].strip()
        elif line.startswith("FORMULA"):
            result["formula"] = line[12:].strip()
        elif line.startswith("PATHWAY"):
            if "pathway" not in result:
                result["pathway"] = {}
            match = re.match(r"PATHWAY\s+(\S+)\s+(.+)", line)
            if match:
                result["pathway"][match.group(1)] = match.group(2)
        elif line.startswith("GENE"):
            if "gene" not in result:
                result["gene"] = {}
            match = re.match(r"GENE\s+(\S+)\s+(.+)", line)
            if match:
                result["gene"][match.group(1)] = match.group(2)
        elif line.startswith("COMPOUND"):
            if "compound" not in result:
                result["compound"] = {}
            match = re.match(r"COMPOUND\s+(\S+)\s+(.+)", line)
            if match:
                result["compound"][match.group(1)] = match.group(2)
        elif line.startswith("REACTION"):
            if "reaction" not in result:
                result["reaction"] = {}
            match = re.match(r"REACTION\s+(\S+)\s+(.+)", line)
            if match:
                result["reaction"][match.group(1)] = match.group(2)
        elif line.startswith("ORTHOLOGY"):
            if "orthology" not in result:
                result["orthology"] = {}
            match = re.match(r"ORTHOLOGY\s+(\S+)\s+(.+)", line)
            if match:
                result["orthology"][match.group(1)] = match.group(2)
        elif line.startswith("DBLINKS"):
            if "dblinks" not in result:
                result["dblinks"] = {}
            match = re.match(r"DBLINKS\s+(\S+):\s+(.+)", line)
            if match:
                result["dblinks"][match.group(1)] = match.group(2).split()
        elif line.startswith("///"):
            break

    return result


def parse_kegg_list(data: str) -> dict[str, str]:
    """Parse KEGG list data into dictionary."""
    result: dict[str, str] = {}
    lines = [line for line in data.split("\n") if line.strip()]

    for line in lines:
        tab_index = line.find("\t")
        if tab_index > 0:
            entry_id = line[:tab_index]
            name = line[tab_index + 1 :]
            result[entry_id] = name

    return result


# Database Information & Statistics
@mcp.tool()
async def get_database_info(database: str) -> str:
    """Get release information and statistics for any KEGG database.
    
    Args:
        database: Database name (kegg, pathway, brite, module, ko, genes, genome, 
                  compound, glycan, reaction, rclass, enzyme, network, variant, 
                  disease, drug, dgroup, or organism code)
    
    Returns:
        JSON string with database information
    """
    try:
        response = await http_client.get(f"/info/{database}")
        return f'{{"database": "{database}", "info": {repr(response.text)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get database info: {str(e)}"}}'


@mcp.tool()
async def list_organisms(limit: int = 100) -> str:
    """Get all KEGG organisms with codes and names.
    
    Args:
        limit: Maximum number of organisms to return (default: 100, max: 1000)
    
    Returns:
        JSON string with organism list
    """
    try:
        limit = min(max(1, limit), 1000)
        response = await http_client.get("/list/organism")
        organisms = parse_kegg_list(response.text)
        
        limited_organisms = dict(list(organisms.items())[:limit])
        
        return f'{{"total_organisms": {len(organisms)}, "returned_count": {len(limited_organisms)}, "organisms": {repr(limited_organisms)}}}'
    except Exception as e:
        return f'{{"error": "Failed to list organisms: {str(e)}"}}'


# Pathway Analysis
@mcp.tool()
async def search_pathways(
    query: str, organism_code: str | None = None, max_results: int = 50
) -> str:
    """Search pathways by keywords or pathway names.
    
    Args:
        query: Search query (pathway name, keywords, or description)
        organism_code: Organism code to filter results (optional, e.g., hsa, mmu, eco)
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        database = f"pathway/{organism_code}" if organism_code else "pathway"
        response = await http_client.get(f"/find/{database}/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "organism_code": {repr(organism_code)}, "total_results": {len(results)}, "returned_count": {len(limited_results)}, "results": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search pathways: {str(e)}"}}'


@mcp.tool()
async def get_pathway_info(pathway_id: str, format: str = "json") -> str:
    """Get detailed information for a specific pathway.
    
    Args:
        pathway_id: Pathway ID (e.g., map00010, hsa00010, ko00010)
        format: Output format (json, kgml, image, conf) - default: json
    
    Returns:
        JSON string with pathway information
    """
    try:
        if format == "json":
            response = await http_client.get(f"/get/{pathway_id}")
            pathway_info = parse_kegg_entry(response.text)
            return f'{{"pathway_id": "{pathway_id}", "info": {repr(pathway_info)}}}'
        else:
            response = await http_client.get(f"/get/{pathway_id}", params={"format": format})
            return f'{{"pathway_id": "{pathway_id}", "format": "{format}", "data": {repr(response.text[:1000])}}}'
    except Exception as e:
        return f'{{"error": "Failed to get pathway info: {str(e)}"}}'


@mcp.tool()
async def get_pathway_genes(pathway_id: str) -> str:
    """Get all genes involved in a specific pathway.
    
    Args:
        pathway_id: Pathway ID (e.g., hsa00010, mmu00010)
    
    Returns:
        JSON string with gene list
    """
    try:
        response = await http_client.get(f"/link/genes/{pathway_id}")
        genes = parse_kegg_list(response.text)
        return f'{{"pathway_id": "{pathway_id}", "genes": {repr(genes)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get pathway genes: {str(e)}"}}'


# Gene Analysis
@mcp.tool()
async def search_genes(
    query: str, organism_code: str | None = None, max_results: int = 50
) -> str:
    """Search genes by name, symbol, or keywords.
    
    Args:
        query: Search query (gene name, symbol, or keywords)
        organism_code: Organism code to filter results (optional, e.g., hsa, mmu)
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        database = f"{organism_code}" if organism_code else "genes"
        response = await http_client.get(f"/find/{database}/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "organism_code": {repr(organism_code)}, "total_results": {len(results)}, "returned_count": {len(limited_results)}, "results": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search genes: {str(e)}"}}'


@mcp.tool()
async def get_gene_info(gene_id: str, include_sequences: bool = False) -> str:
    """Get detailed information for a specific gene.
    
    Args:
        gene_id: Gene ID (e.g., hsa:1956, mmu:11651, eco:b0008)
        include_sequences: Include amino acid and nucleotide sequences (default: false)
    
    Returns:
        JSON string with gene information
    """
    try:
        response = await http_client.get(f"/get/{gene_id}")
        gene_info = parse_kegg_entry(response.text)
        
        if include_sequences:
            # Try to get sequences
            try:
                seq_response = await http_client.get(f"/get/{gene_id}/aaseq")
                gene_info["aa_sequence"] = seq_response.text
            except:
                pass
            
            try:
                seq_response = await http_client.get(f"/get/{gene_id}/ntseq")
                gene_info["nt_sequence"] = seq_response.text
            except:
                pass
        
        return f'{{"gene_id": "{gene_id}", "info": {repr(gene_info)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get gene info: {str(e)}"}}'


# Compound Analysis
@mcp.tool()
async def search_compounds(
    query: str, search_type: str = "name", max_results: int = 50
) -> str:
    """Search compounds by name, formula, or chemical structure.
    
    Args:
        query: Search query (compound name, formula, or identifier)
        search_type: Type of search (name, formula, exact_mass, mol_weight) - default: name
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        response = await http_client.get(f"/find/compound/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "search_type": "{search_type}", "total_results": {len(results)}, "returned_count": {len(limited_results)}, "results": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search compounds: {str(e)}"}}'


@mcp.tool()
async def get_compound_info(compound_id: str) -> str:
    """Get detailed information for a specific compound.
    
    Args:
        compound_id: Compound ID (e.g., C00002, C00031, cpd:C00002)
    
    Returns:
        JSON string with compound information
    """
    try:
        # Remove 'cpd:' prefix if present
        clean_id = compound_id.replace("cpd:", "")
        response = await http_client.get(f"/get/{clean_id}")
        compound_info = parse_kegg_entry(response.text)
        return f'{{"compound_id": "{compound_id}", "info": {repr(compound_info)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get compound info: {str(e)}"}}'


# Reaction & Enzyme Analysis
@mcp.tool()
async def search_reactions(query: str, max_results: int = 50) -> str:
    """Search biochemical reactions by keywords or reaction components.
    
    Args:
        query: Search query (reaction name, enzyme, or compound)
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        response = await http_client.get(f"/find/reaction/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "total_found": {len(results)}, "returned_count": {len(limited_results)}, "reactions": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search reactions: {str(e)}"}}'


@mcp.tool()
async def get_reaction_info(reaction_id: str) -> str:
    """Get detailed information for a specific reaction.
    
    Args:
        reaction_id: Reaction ID (e.g., R00001, R00002)
    
    Returns:
        JSON string with reaction information
    """
    try:
        response = await http_client.get(f"/get/{reaction_id}")
        reaction_info = parse_kegg_entry(response.text)
        return f'{{"reaction_id": "{reaction_id}", "info": {repr(reaction_info)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get reaction info: {str(e)}"}}'


@mcp.tool()
async def search_enzymes(query: str, max_results: int = 50) -> str:
    """Search enzymes by EC number or enzyme name.
    
    Args:
        query: Search query (EC number or enzyme name)
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        response = await http_client.get(f"/find/enzyme/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "total_found": {len(results)}, "returned_count": {len(limited_results)}, "enzymes": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search enzymes: {str(e)}"}}'


@mcp.tool()
async def get_enzyme_info(ec_number: str) -> str:
    """Get detailed enzyme information by EC number.
    
    Args:
        ec_number: EC number (e.g., ec:1.1.1.1)
    
    Returns:
        JSON string with enzyme information
    """
    try:
        # Remove 'ec:' prefix if present
        clean_ec = ec_number.replace("ec:", "")
        response = await http_client.get(f"/get/{clean_ec}")
        enzyme_info = parse_kegg_entry(response.text)
        return f'{{"ec_number": "{ec_number}", "info": {repr(enzyme_info)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get enzyme info: {str(e)}"}}'


# Disease & Drug Analysis
@mcp.tool()
async def search_diseases(query: str, max_results: int = 50) -> str:
    """Search human diseases by name or keywords.
    
    Args:
        query: Search query (disease name or keywords)
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        response = await http_client.get(f"/find/disease/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "total_found": {len(results)}, "returned_count": {len(limited_results)}, "diseases": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search diseases: {str(e)}"}}'


@mcp.tool()
async def get_disease_info(disease_id: str) -> str:
    """Get detailed information for a specific disease.
    
    Args:
        disease_id: Disease ID (e.g., H00001, H00002)
    
    Returns:
        JSON string with disease information
    """
    try:
        response = await http_client.get(f"/get/{disease_id}")
        disease_info = parse_kegg_entry(response.text)
        return f'{{"disease_id": "{disease_id}", "info": {repr(disease_info)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get disease info: {str(e)}"}}'


@mcp.tool()
async def search_drugs(query: str, max_results: int = 50) -> str:
    """Search drugs by name, target, or indication.
    
    Args:
        query: Search query (drug name, target, or indication)
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        response = await http_client.get(f"/find/drug/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "total_found": {len(results)}, "returned_count": {len(limited_results)}, "drugs": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search drugs: {str(e)}"}}'


@mcp.tool()
async def get_drug_info(drug_id: str) -> str:
    """Get detailed information for a specific drug.
    
    Args:
        drug_id: Drug ID (e.g., D00001, D00002)
    
    Returns:
        JSON string with drug information
    """
    try:
        response = await http_client.get(f"/get/{drug_id}")
        drug_info = parse_kegg_entry(response.text)
        return f'{{"drug_id": "{drug_id}", "info": {repr(drug_info)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get drug info: {str(e)}"}}'


@mcp.tool()
async def get_drug_interactions(drug_ids: list[str]) -> str:
    """Find adverse drug-drug interactions.
    
    Args:
        drug_ids: Drug IDs to check for interactions (1-10)
    
    Returns:
        JSON string with interaction information
    """
    try:
        if not drug_ids or len(drug_ids) == 0 or len(drug_ids) > 10:
            return '{"error": "Drug IDs array must contain 1-10 items"}'
        
        drug_list = "+".join(drug_ids)
        response = await http_client.get(f"/ddi/{drug_list}")
        interactions = parse_kegg_list(response.text)
        
        return f'{{"drug_ids": {repr(drug_ids)}, "interaction_count": {len(interactions)}, "interactions": {repr(interactions)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get drug interactions: {str(e)}"}}'


# Module & Orthology Analysis
@mcp.tool()
async def search_modules(query: str, max_results: int = 50) -> str:
    """Search KEGG modules by name or function.
    
    Args:
        query: Search query (module name or function)
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        response = await http_client.get(f"/find/module/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "total_found": {len(results)}, "returned_count": {len(limited_results)}, "modules": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search modules: {str(e)}"}}'


@mcp.tool()
async def get_module_info(module_id: str) -> str:
    """Get detailed information for a specific module.
    
    Args:
        module_id: Module ID (e.g., M00001, M00002)
    
    Returns:
        JSON string with module information
    """
    try:
        response = await http_client.get(f"/get/{module_id}")
        module_info = parse_kegg_entry(response.text)
        return f'{{"module_id": "{module_id}", "info": {repr(module_info)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get module info: {str(e)}"}}'


@mcp.tool()
async def search_ko_entries(query: str, max_results: int = 50) -> str:
    """Search KEGG Orthology entries by function or gene name.
    
    Args:
        query: Search query (function or gene name)
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        response = await http_client.get(f"/find/ko/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "total_found": {len(results)}, "returned_count": {len(limited_results)}, "ko_entries": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search KO entries: {str(e)}"}}'


@mcp.tool()
async def get_ko_info(ko_id: str) -> str:
    """Get detailed information for a specific KO entry.
    
    Args:
        ko_id: KO ID (e.g., K00001, K00002)
    
    Returns:
        JSON string with KO information
    """
    try:
        response = await http_client.get(f"/get/{ko_id}")
        ko_info = parse_kegg_entry(response.text)
        return f'{{"ko_id": "{ko_id}", "info": {repr(ko_info)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get KO info: {str(e)}"}}'


# Glycan Analysis
@mcp.tool()
async def search_glycans(query: str, max_results: int = 50) -> str:
    """Search glycan structures by name or composition.
    
    Args:
        query: Search query (glycan name or composition)
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        response = await http_client.get(f"/find/glycan/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "total_found": {len(results)}, "returned_count": {len(limited_results)}, "glycans": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search glycans: {str(e)}"}}'


@mcp.tool()
async def get_glycan_info(glycan_id: str) -> str:
    """Get detailed information for a specific glycan.
    
    Args:
        glycan_id: Glycan ID (e.g., G00001, G00002)
    
    Returns:
        JSON string with glycan information
    """
    try:
        response = await http_client.get(f"/get/{glycan_id}")
        glycan_info = parse_kegg_entry(response.text)
        return f'{{"glycan_id": "{glycan_id}", "info": {repr(glycan_info)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get glycan info: {str(e)}"}}'


# BRITE Hierarchy Analysis
@mcp.tool()
async def search_brite(
    query: str, hierarchy_type: str = "br", max_results: int = 50
) -> str:
    """Search BRITE functional hierarchies.
    
    Args:
        query: Search query (function or category)
        hierarchy_type: Type of BRITE hierarchy (br, ko, jp) - default: br
        max_results: Maximum number of results (1-1000, default: 50)
    
    Returns:
        JSON string with search results
    """
    try:
        max_results = min(max(1, max_results), 1000)
        response = await http_client.get(f"/find/brite/{query}")
        results = parse_kegg_list(response.text)
        
        limited_results = dict(list(results.items())[:max_results])
        
        return f'{{"query": "{query}", "hierarchy_type": "{hierarchy_type}", "total_found": {len(results)}, "returned_count": {len(limited_results)}, "brite_entries": {repr(limited_results)}}}'
    except Exception as e:
        return f'{{"error": "Failed to search BRITE: {str(e)}"}}'


@mcp.tool()
async def get_brite_info(brite_id: str, format: str = "json") -> str:
    """Get detailed information for a specific BRITE entry.
    
    Args:
        brite_id: BRITE ID (e.g., br:br08301, ko:K00001)
        format: Output format (json, htext) - default: json
    
    Returns:
        JSON string with BRITE information
    """
    try:
        # Remove prefix if present
        clean_id = brite_id.replace("br:", "").replace("ko:", "")
        
        if format == "htext":
            endpoint = f"/get/{clean_id}/htext"
        else:
            endpoint = f"/get/{clean_id}"
        
        response = await http_client.get(endpoint)
        
        if format == "json":
            brite_info = parse_kegg_entry(response.text)
            return f'{{"brite_id": "{brite_id}", "info": {repr(brite_info)}}}'
        else:
            return f'{{"brite_id": "{brite_id}", "format": "{format}", "data": {repr(response.text[:2000])}}}'
    except Exception as e:
        return f'{{"error": "Failed to get BRITE info: {str(e)}"}}'


# Advanced Analysis Tools
@mcp.tool()
async def get_pathway_compounds(pathway_id: str) -> str:
    """Get all compounds involved in a specific pathway.
    
    Args:
        pathway_id: Pathway ID (e.g., map00010, hsa00010)
    
    Returns:
        JSON string with compound list
    """
    try:
        response = await http_client.get(f"/link/compound/{pathway_id}")
        compounds = parse_kegg_list(response.text)
        return f'{{"pathway_id": "{pathway_id}", "compound_count": {len(compounds)}, "compounds": {repr(compounds)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get pathway compounds: {str(e)}"}}'


@mcp.tool()
async def get_pathway_reactions(pathway_id: str) -> str:
    """Get all reactions involved in a specific pathway.
    
    Args:
        pathway_id: Pathway ID (e.g., map00010, rn00010)
    
    Returns:
        JSON string with reaction list
    """
    try:
        response = await http_client.get(f"/link/reaction/{pathway_id}")
        reactions = parse_kegg_list(response.text)
        return f'{{"pathway_id": "{pathway_id}", "reaction_count": {len(reactions)}, "reactions": {repr(reactions)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get pathway reactions: {str(e)}"}}'


@mcp.tool()
async def get_compound_reactions(compound_id: str) -> str:
    """Get all reactions involving a specific compound.
    
    Args:
        compound_id: Compound ID (e.g., C00002, C00031)
    
    Returns:
        JSON string with reaction list
    """
    try:
        clean_id = compound_id.replace("cpd:", "")
        response = await http_client.get(f"/link/reaction/{clean_id}")
        reactions = parse_kegg_list(response.text)
        return f'{{"compound_id": "{compound_id}", "reaction_count": {len(reactions)}, "reactions": {repr(reactions)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get compound reactions: {str(e)}"}}'


@mcp.tool()
async def get_gene_orthologs(
    gene_id: str, target_organisms: list[str] | None = None
) -> str:
    """Find orthologous genes across organisms.
    
    Args:
        gene_id: Gene ID (e.g., hsa:1956)
        target_organisms: Target organism codes (optional, e.g., [mmu, rno, dme])
    
    Returns:
        JSON string with ortholog information
    """
    try:
        # Get KO links for the gene
        response = await http_client.get(f"/link/ko/{gene_id}")
        ko_links = parse_kegg_list(response.text)
        
        orthologs: dict[str, str] = {}
        
        if target_organisms:
            # Get genes for each KO in target organisms
            for ko in ko_links.keys():
                for org in target_organisms:
                    try:
                        org_response = await http_client.get(f"/link/{org}/{ko}")
                        org_genes = parse_kegg_list(org_response.text)
                        orthologs.update(org_genes)
                    except:
                        # Continue if organism doesn't have this KO
                        pass
        else:
            # Get all orthologs
            for ko in ko_links.keys():
                try:
                    ko_response = await http_client.get(f"/link/genes/{ko}")
                    all_genes = parse_kegg_list(ko_response.text)
                    orthologs.update(all_genes)
                except:
                    pass
        
        return f'{{"gene_id": "{gene_id}", "target_organisms": {repr(target_organisms)}, "ortholog_count": {len(orthologs)}, "orthologs": {repr(orthologs)}}}'
    except Exception as e:
        return f'{{"error": "Failed to get gene orthologs: {str(e)}"}}'


@mcp.tool()
async def batch_entry_lookup(
    entry_ids: list[str], operation: str = "info"
) -> str:
    """Process multiple KEGG entries efficiently.
    
    Args:
        entry_ids: KEGG entry IDs (1-50)
        operation: Operation to perform (info, sequence, pathway, link) - default: info
    
    Returns:
        JSON string with batch results
    """
    try:
        if not entry_ids or len(entry_ids) == 0 or len(entry_ids) > 50:
            return '{"error": "Entry IDs array must contain 1-50 items"}'
        
        results = []
        
        for entry_id in entry_ids:
            try:
                if operation == "sequence":
                    response = await http_client.get(f"/get/{entry_id}/aaseq")
                    results.append(
                        {
                            "entry_id": entry_id,
                            "data": response.text[:1000],
                            "success": True,
                        }
                    )
                elif operation == "pathway":
                    response = await http_client.get(f"/link/pathway/{entry_id}")
                    link_data = parse_kegg_list(response.text)
                    results.append(
                        {"entry_id": entry_id, "data": link_data, "success": True}
                    )
                elif operation == "link":
                    response = await http_client.get(f"/link/ko/{entry_id}")
                    link_data = parse_kegg_list(response.text)
                    results.append(
                        {"entry_id": entry_id, "data": link_data, "success": True}
                    )
                else:  # info
                    response = await http_client.get(f"/get/{entry_id}")
                    entry_info = parse_kegg_entry(response.text)
                    results.append(
                        {"entry_id": entry_id, "data": entry_info, "success": True}
                    )
            except Exception as e:
                results.append(
                    {
                        "entry_id": entry_id,
                        "error": str(e),
                        "success": False,
                    }
                )
        
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        
        return f'{{"operation": "{operation}", "total_entries": {len(entry_ids)}, "successful": {successful}, "failed": {failed}, "results": {repr(results)}}}'
    except Exception as e:
        return f'{{"error": "Batch lookup failed: {str(e)}"}}'


# Cross-References & Integration
@mcp.tool()
async def convert_identifiers(
    source_db: str, target_db: str, identifiers: list[str] | None = None
) -> str:
    """Convert between KEGG and external database identifiers.
    
    Args:
        source_db: Source database (e.g., hsa, ncbi-geneid, uniprot)
        target_db: Target database (e.g., hsa, ncbi-geneid, uniprot)
        identifiers: Identifiers to convert (optional, for batch conversion)
    
    Returns:
        JSON string with conversion results
    """
    try:
        if identifiers and len(identifiers) > 0:
            # Convert specific identifiers
            identifier_list = "+".join(identifiers)
            endpoint = f"/conv/{target_db}/{identifier_list}"
        else:
            # Get all conversions between databases
            endpoint = f"/conv/{target_db}/{source_db}"
        
        response = await http_client.get(endpoint)
        conversions = parse_kegg_list(response.text)
        
        return f'{{"source_db": "{source_db}", "target_db": "{target_db}", "conversion_count": {len(conversions)}, "conversions": {repr(conversions)}}}'
    except Exception as e:
        return f'{{"error": "Failed to convert identifiers: {str(e)}"}}'


@mcp.tool()
async def find_related_entries(
    source_db: str, target_db: str, source_entries: list[str] | None = None
) -> str:
    """Find related entries across KEGG databases using cross-references.
    
    Args:
        source_db: Source database (e.g., pathway, compound, gene)
        target_db: Target database (e.g., pathway, compound, gene)
        source_entries: Source entries to find links for (optional)
    
    Returns:
        JSON string with related entries
    """
    try:
        if source_entries and len(source_entries) > 0:
            # Find links for specific entries
            entry_list = "+".join(source_entries)
            endpoint = f"/link/{target_db}/{entry_list}"
        else:
            # Get all links between databases
            endpoint = f"/link/{target_db}/{source_db}"
        
        response = await http_client.get(endpoint)
        links = parse_kegg_list(response.text)
        
        return f'{{"source_db": "{source_db}", "target_db": "{target_db}", "link_count": {len(links)}, "links": {repr(links)}}}'
    except Exception as e:
        return f'{{"error": "Failed to find related entries: {str(e)}"}}'


async def main():
    """Run the MCP server with Streamable HTTP transport."""
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("MCP_HOST", "localhost")
    port = int(os.getenv("MCP_PORT", "3000"))
    path = os.getenv("MCP_PATH", "/mcp")
    
    # Configure FastMCP for Streamable HTTP
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.streamable_http_path = path
    
    print(f"Starting KEGG MCP Server on http://{host}:{port}{path}")
    print("Transport: Streamable HTTP")
    
    # Run the server
    await mcp.run_streamable_http_async()


if __name__ == "__main__":
    asyncio.run(main())
