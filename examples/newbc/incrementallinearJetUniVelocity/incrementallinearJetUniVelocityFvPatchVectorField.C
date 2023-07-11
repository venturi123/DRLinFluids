/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "incrementallinearJetUniVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

Foam::scalar Foam::incrementallinearJetUniVelocityFvPatchVectorField::t() const
{
    return db().time().timeOutputValue();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::incrementallinearJetUniVelocityFvPatchVectorField::
incrementallinearJetUniVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(p, iF),
    alpha_(0.0),
    t0_(0.0),
    deltat_(0.0),
    v0_(Zero),
    v1_(Zero)
{
}


Foam::incrementallinearJetUniVelocityFvPatchVectorField::
incrementallinearJetUniVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchVectorField(p, iF),
    alpha_(dict.lookup<scalar>("alpha")),
    t0_(dict.lookup<scalar>("t0")),
    deltat_(dict.lookup<scalar>("deltaT")),
    v0_(pTraits<vector>(dict.lookup("v0"))),
    v1_(pTraits<vector>(dict.lookup("v1")))
{


    fixedValueFvPatchVectorField::evaluate();

    /*
    // Initialise with the value entry if evaluation is not possible
    fvPatchVectorField::operator=
    (
        vectorField("value", dict, p.size())
    );
    */
}


Foam::incrementallinearJetUniVelocityFvPatchVectorField::
incrementallinearJetUniVelocityFvPatchVectorField
(
    const incrementallinearJetUniVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchVectorField(ptf, p, iF, mapper),
    alpha_(ptf.alpha_),
    t0_(ptf.t0_),
    deltat_(ptf.deltat_),
    v0_(ptf.v0_),
    v1_(ptf.v1_)
{}


Foam::incrementallinearJetUniVelocityFvPatchVectorField::
incrementallinearJetUniVelocityFvPatchVectorField
(
    const incrementallinearJetUniVelocityFvPatchVectorField& ptf
)
:
    fixedValueFvPatchVectorField(ptf),
    alpha_(ptf.alpha_),
    t0_(ptf.t0_),
    deltat_(ptf.deltat_),
    v0_(ptf.v0_),
    v1_(ptf.v1_)
{}


Foam::incrementallinearJetUniVelocityFvPatchVectorField::
incrementallinearJetUniVelocityFvPatchVectorField
(
    const incrementallinearJetUniVelocityFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(ptf, iF),
    alpha_(ptf.alpha_),
    t0_(ptf.t0_),
    deltat_(ptf.deltat_),
    v0_(ptf.v0_),
    v1_(ptf.v1_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::incrementallinearJetUniVelocityFvPatchVectorField::autoMap
(
    const fvPatchFieldMapper& m
)
{
    fixedValueFvPatchVectorField::autoMap(m);
    // m(fieldData_, fieldData_);
}


void Foam::incrementallinearJetUniVelocityFvPatchVectorField::rmap
(
    const fvPatchVectorField& ptf,
    const labelList& addr
)
{
    fixedValueFvPatchVectorField::rmap(ptf, addr);

    // const incrementallinearJetUniVelocityFvPatchVectorField& tiptf =
    //     refCast<const incrementallinearJetUniVelocityFvPatchVectorField>(ptf);

    // fieldData_.rmap(tiptf.fieldData_, addr);
}


void Foam::incrementallinearJetUniVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // vector v_ = (v0_ - v1_) * pow(1 - alpha_, (t() - t0_) / deltat_ + 1) + v1_;
	vector v_ = v0_ + (v1_ - v0_) / ((t() - t0_+ deltat_) / deltat_);

    fixedValueFvPatchVectorField::operator==
    (
        v_
    );


    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::incrementallinearJetUniVelocityFvPatchVectorField::write
(
    Ostream& os
) const
{
    fvPatchVectorField::write(os);
    writeEntry(os, "alpha", alpha_);
    writeEntry(os, "t0", t0_);
    writeEntry(os, "deltaT", deltat_);
    writeEntry(os, "v0", v0_);
    writeEntry(os, "v1", v1_);
}


// * * * * * * * * * * * * * * Build Macro Function  * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
		incrementallinearJetUniVelocityFvPatchVectorField
    );
}

// ************************************************************************* //
